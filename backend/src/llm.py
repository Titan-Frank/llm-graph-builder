import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
import os
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_experimental.graph_transformers.llm import _Graph
from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks
from langchain_aws import ChatBedrock
from langchain_community.chat_models import ChatOllama
import boto3
import google.auth
from src.shared.constants import (
    ADDITIONAL_INSTRUCTIONS,
    DESCRIPTION_SUMMARIZATION_SYSTEM_PROMPT,
    EDUCATION_ADDITIONAL_INSTRUCTIONS,
    EDUCATION_SCHEMA_PRESET,
    EDUCATION_SECOND_PASS_ADDITIONAL_INSTRUCTIONS,
    ENTITY_DESCRIPTION_SUMMARIZATION_PROMPT,
    RELATIONSHIP_DESCRIPTION_SUMMARIZATION_PROMPT,
    SECOND_PASS_ADDITIONAL_INSTRUCTIONS,
)
from src.shared.llm_graph_builder_exception import LLMGraphBuilderException
import re
from typing import Dict, List, Optional, Set, Tuple
from langchain_core.callbacks.manager import CallbackManager
from src.shared.common_fn import UniversalTokenUsageHandler,get_value_from_env

EDUCATION_SCHEMA_PROFILE = EDUCATION_SCHEMA_PRESET["name"]

def get_llm(model: str):
    """Retrieve the specified language model based on the model name."""
    raw_model = model.strip()
    model = raw_model.upper()
    normalized_model = re.sub(r"[^A-Z0-9]+", "_", model).strip("_")
    candidate_env_keys = [
        f"LLM_MODEL_CONFIG_{normalized_model}",
        f"LLM_MODEL_CONFIG_{raw_model}",
    ]
    env_key = candidate_env_keys[0]
    env_value = next((os.getenv(key) for key in candidate_env_keys if os.getenv(key) not in (None, "")), None)

    if not env_value:
        err = f"Environment variable not found for model '{raw_model}'. Tried: {', '.join(candidate_env_keys)}"
        logging.error(err)
        raise Exception(err)
    
    logging.info("Model: {}".format(env_key))
    callback_handler = UniversalTokenUsageHandler()
    callback_manager = CallbackManager([callback_handler])
    try:
        if "GEMINI" in model:
            model_name = env_value
            credentials, project_id = google.auth.default()
            llm = ChatVertexAI(
                model_name=model_name,
                credentials=credentials,
                project=project_id,
                temperature=0,
                callbacks=callback_manager,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                },
            
            )
        elif "OPENAI" in model:
            model_name, api_key = env_value.split(",")
            if "MINI" in model:
                llm= ChatOpenAI(
                api_key=api_key,
                model=model_name,
                callbacks=callback_manager,
                )
            else:
                llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=0,
                callbacks=callback_manager,
                )

        elif "AZURE" in model:
            model_name, api_endpoint, api_key, api_version = env_value.split(",")
            llm = AzureChatOpenAI(
                api_key=api_key,
                azure_endpoint=api_endpoint,
                azure_deployment=model_name,  # takes precedence over model parameter
                api_version=api_version,
                temperature=0,
                max_tokens=None,
                timeout=None,
                callbacks=callback_manager,
            )

        elif "ANTHROPIC" in model:
            model_name, api_key = env_value.split(",")
            llm = ChatAnthropic(
                api_key=api_key, model=model_name, temperature=0, timeout=None,callbacks=callback_manager, 
            )

        elif "FIREWORKS" in model:
            model_name, api_key = env_value.split(",")
            llm = ChatFireworks(api_key=api_key, model=model_name,callbacks=callback_manager)

        elif "GROQ" in model:
            model_name, base_url, api_key = env_value.split(",")
            llm = ChatGroq(api_key=api_key, model_name=model_name, temperature=0,callbacks=callback_manager)

        elif "BEDROCK" in model:
            model_name, aws_access_key, aws_secret_key, region_name = env_value.split(",")
            bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
            )

            llm = ChatBedrock(
                client=bedrock_client,region_name=region_name, model_id=model_name, model_kwargs=dict(temperature=0),callbacks=callback_manager, 
            )

        elif "OLLAMA" in model:
            model_name, base_url = env_value.split(",")
            llm = ChatOllama(base_url=base_url, model=model_name,callbacks=callback_manager)

        elif "DIFFBOT" in model:
            #model_name = "diffbot"
            model_name, api_key = env_value.split(",")
            llm = DiffbotGraphTransformer(
                diffbot_api_key=api_key,
                extract_types=["entities", "facts"],
            )
            callback_handler = None
        
        else: 
            model_name, api_endpoint, api_key = env_value.split(",")
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=api_endpoint,
                model=model_name,
                temperature=0,
                callbacks=callback_manager,
            )
    except Exception as e:
        err = f"Error while creating LLM '{model}': {str(e)}"
        logging.error(err)
        raise Exception(err)
 
    logging.info(f"Model created - Model Version: {model}")
    return llm, model_name, callback_handler

def get_llm_model_name(llm):
    """Extract name of llm model from llm object"""
    for attr in ["model_name", "model", "model_id"]:
        model_name = getattr(llm, attr, None)
        if model_name:
            return model_name.lower()
    logging.info("Could not determine model name; defaulting to empty string")
    return ""

def get_combined_chunks(chunkId_chunkDoc_list, chunks_to_combine):
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={"combined_chunk_ids": combined_chunks_ids[i]},
            )
        )
    return combined_chunk_document_list


def parse_allowed_nodes(allowed_nodes: Optional[str]) -> List[str]:
    if not allowed_nodes:
        return []
    return [node.strip() for node in allowed_nodes.split(",") if node.strip()]


def parse_allowed_relationships(
    allowed_relationships: Optional[str],
    allowed_nodes: List[str],
) -> List[Tuple[str, str, str]]:
    if not allowed_relationships:
        return []

    parsed_nodes = {node.strip() for node in allowed_nodes}
    items = [item.strip() for item in allowed_relationships.split(",") if item.strip()]
    if len(items) % 3 != 0:
        raise LLMGraphBuilderException("allowedRelationship must be a multiple of 3 (source, relationship, target)")

    parsed_relationships: List[Tuple[str, str, str]] = []
    for i in range(0, len(items), 3):
        source, relation, target = items[i:i + 3]
        if parsed_nodes and (source not in parsed_nodes or target not in parsed_nodes):
            raise LLMGraphBuilderException(
                f"Invalid relationship ({source}, {relation}, {target}): source or target not in allowedNodes"
            )
        parsed_relationships.append((source, relation, target))
    return parsed_relationships


def serialize_allowed_relationships(allowed_relationships: List[Tuple[str, str, str]]) -> str:
    return ",".join(item for relationship in allowed_relationships for item in relationship)


def prepend_instruction_block(base_instructions: Optional[str], prefix: str) -> str:
    if base_instructions:
        return prefix + "\n" + base_instructions
    return prefix


def infer_schema_profile_from_allowed_nodes(allowed_nodes: Optional[str]) -> Optional[str]:
    parsed_allowed_nodes = {
        normalize_graph_key(node)
        for node in parse_allowed_nodes(allowed_nodes)
    }
    education_nodes = {
        normalize_graph_key(node)
        for node in EDUCATION_SCHEMA_PRESET["allowed_nodes"]
    }
    if len(parsed_allowed_nodes & education_nodes) >= 3:
        return EDUCATION_SCHEMA_PROFILE
    return None


def resolve_schema_profile(
    allowed_nodes: Optional[str],
    allowed_relationships: Optional[str],
    additional_instructions: Optional[str],
    schema_profile: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    normalized_profile = schema_profile.strip().lower() if schema_profile else infer_schema_profile_from_allowed_nodes(allowed_nodes)
    if not normalized_profile:
        return allowed_nodes, allowed_relationships, additional_instructions, None

    if normalized_profile != EDUCATION_SCHEMA_PROFILE:
        raise LLMGraphBuilderException(f"Unsupported schema profile '{schema_profile}'")

    resolved_nodes = allowed_nodes or ",".join(EDUCATION_SCHEMA_PRESET["allowed_nodes"])
    resolved_relationships = allowed_relationships or serialize_allowed_relationships(
        EDUCATION_SCHEMA_PRESET["allowed_relationships"]
    )
    resolved_instructions = prepend_instruction_block(additional_instructions, EDUCATION_ADDITIONAL_INSTRUCTIONS)
    return resolved_nodes, resolved_relationships, resolved_instructions, normalized_profile

def get_chunk_id_as_doc_metadata(chunkId_chunkDoc_list):
    combined_chunk_document_list = [
       Document(
           page_content=document["chunk_doc"].page_content,
           metadata={"chunk_id": [document["chunk_id"]]},
       )
       for document in chunkId_chunkDoc_list
   ]
    return combined_chunk_document_list


def build_graph_transformer(llm, allowedNodes, allowedRelationship, additional_instructions, use_structured_output):
    return LLMGraphTransformer(
        llm=llm,
        node_properties=["description"] if use_structured_output else False,
        relationship_properties=["description", "strength"] if use_structured_output else False,
        allowed_nodes=allowedNodes,
        allowed_relationships=allowedRelationship,
        ignore_tool_usage=not use_structured_output,
        additional_instructions=ADDITIONAL_INSTRUCTIONS + (additional_instructions if additional_instructions else ""),
    )


def is_structured_output_validation_error(error: Exception) -> bool:
    error_message = str(error).lower()
    return (
        "dynamicgraph" in error_message
        and any(
            marker in error_message
            for marker in ["validation error", "validation errors", "invalid json", "json_invalid", "field required"]
        )
    )


def count_graph_elements(graph_document_list: List[GraphDocument]) -> Tuple[int, int]:
    node_count = sum(len(graph_document.nodes) for graph_document in graph_document_list)
    relationship_count = sum(len(graph_document.relationships) for graph_document in graph_document_list)
    return node_count, relationship_count


def normalize_graph_key(value) -> str:
    return str(value).strip().lower()


def merge_property_value(existing_value, incoming_value):
    if incoming_value in (None, "", []):
        return existing_value
    if existing_value in (None, "", []):
        return incoming_value

    if isinstance(existing_value, str) and isinstance(incoming_value, str):
        if existing_value == incoming_value:
            return existing_value
        if existing_value in incoming_value:
            return incoming_value
        if incoming_value in existing_value:
            return existing_value
        return incoming_value if len(incoming_value) > len(existing_value) else existing_value

    if isinstance(existing_value, (int, float)) and isinstance(incoming_value, (int, float)):
        return incoming_value if incoming_value > existing_value else existing_value

    return existing_value


def merge_properties(existing_properties: dict, incoming_properties: dict) -> dict:
    merged_properties = dict(existing_properties or {})
    for key, value in (incoming_properties or {}).items():
        merged_properties[key] = merge_property_value(merged_properties.get(key), value)
    return merged_properties


def merge_nodes(existing_node: Node, incoming_node: Node) -> Node:
    return Node(
        id=existing_node.id,
        type=existing_node.type,
        properties=merge_properties(existing_node.properties, incoming_node.properties),
    )


def merge_relationships(existing_relationship: Relationship, incoming_relationship: Relationship) -> Relationship:
    return Relationship(
        source=merge_nodes(existing_relationship.source, incoming_relationship.source),
        target=merge_nodes(existing_relationship.target, incoming_relationship.target),
        type=existing_relationship.type,
        properties=merge_properties(existing_relationship.properties, incoming_relationship.properties),
    )


def merge_graph_document_lists(
    primary_graph_documents: List[GraphDocument],
    secondary_graph_documents: List[GraphDocument],
) -> List[GraphDocument]:
    merged_documents: Dict[Tuple[str, ...], GraphDocument] = {}
    ordered_keys: List[Tuple[str, ...]] = []

    def get_source_key(graph_document: GraphDocument) -> Tuple[str, ...]:
        metadata = graph_document.source.metadata or {}
        source_ids = metadata.get("combined_chunk_ids") or metadata.get("chunk_id") or []
        if not source_ids:
            source_ids = [graph_document.source.page_content[:100]]
        return tuple(str(item) for item in source_ids)

    def upsert_graph_document(graph_document: GraphDocument):
        source_key = get_source_key(graph_document)
        if source_key not in merged_documents:
            merged_documents[source_key] = GraphDocument(
                nodes=[],
                relationships=[],
                source=graph_document.source,
            )
            ordered_keys.append(source_key)

        merged_document = merged_documents[source_key]

        existing_nodes = {
            (normalize_graph_key(node.type), normalize_graph_key(node.id)): node
            for node in merged_document.nodes
        }
        for node in graph_document.nodes:
            node_key = (normalize_graph_key(node.type), normalize_graph_key(node.id))
            if node_key in existing_nodes:
                existing_nodes[node_key] = merge_nodes(existing_nodes[node_key], node)
            else:
                existing_nodes[node_key] = node

        existing_relationships = {
            (
                normalize_graph_key(rel.source.type),
                normalize_graph_key(rel.source.id),
                normalize_graph_key(rel.type),
                normalize_graph_key(rel.target.type),
                normalize_graph_key(rel.target.id),
            ): rel
            for rel in merged_document.relationships
        }
        for relationship in graph_document.relationships:
            relationship_key = (
                normalize_graph_key(relationship.source.type),
                normalize_graph_key(relationship.source.id),
                normalize_graph_key(relationship.type),
                normalize_graph_key(relationship.target.type),
                normalize_graph_key(relationship.target.id),
            )
            if relationship_key in existing_relationships:
                existing_relationships[relationship_key] = merge_relationships(
                    existing_relationships[relationship_key],
                    relationship,
                )
            else:
                existing_relationships[relationship_key] = relationship

        merged_document.nodes = list(existing_nodes.values())
        merged_document.relationships = list(existing_relationships.values())

    for graph_document in primary_graph_documents:
        upsert_graph_document(graph_document)
    for graph_document in secondary_graph_documents:
        upsert_graph_document(graph_document)

    return [merged_documents[source_key] for source_key in ordered_keys]


def build_allowed_node_type_set(allowed_nodes: List[str]) -> Set[str]:
    return {normalize_graph_key(node) for node in allowed_nodes}


def build_allowed_relationship_type_set(
    allowed_relationships: List[Tuple[str, str, str]],
) -> Set[Tuple[str, str, str]]:
    return {
        (
            normalize_graph_key(source),
            normalize_graph_key(relation),
            normalize_graph_key(target),
        )
        for source, relation, target in allowed_relationships
    }


def filter_graph_documents_by_schema(
    graph_document_list: List[GraphDocument],
    allowed_nodes: List[str],
    allowed_relationships: List[Tuple[str, str, str]],
) -> List[GraphDocument]:
    if not allowed_nodes and not allowed_relationships:
        return graph_document_list

    allowed_node_types = build_allowed_node_type_set(allowed_nodes)
    allowed_relationship_types = build_allowed_relationship_type_set(allowed_relationships)
    dropped_nodes = 0
    dropped_relationships = 0
    filtered_documents: List[GraphDocument] = []

    for graph_document in graph_document_list:
        filtered_nodes: Dict[Tuple[str, str], Node] = {}
        filtered_relationships: Dict[Tuple[str, str, str, str, str], Relationship] = {}

        for node in graph_document.nodes:
            node_type = normalize_graph_key(node.type)
            node_id = normalize_graph_key(node.id)
            if allowed_node_types and node_type not in allowed_node_types:
                dropped_nodes += 1
                continue
            filtered_nodes[(node_type, node_id)] = node

        for relationship in graph_document.relationships:
            source = getattr(relationship, "source", None)
            target = getattr(relationship, "target", None)
            if source is None or target is None:
                dropped_relationships += 1
                continue

            source_type = normalize_graph_key(source.type)
            source_id = normalize_graph_key(source.id)
            relationship_type = normalize_graph_key(relationship.type)
            target_type = normalize_graph_key(target.type)
            target_id = normalize_graph_key(target.id)

            if allowed_node_types and (source_type not in allowed_node_types or target_type not in allowed_node_types):
                dropped_relationships += 1
                continue
            if allowed_relationship_types and (
                source_type,
                relationship_type,
                target_type,
            ) not in allowed_relationship_types:
                dropped_relationships += 1
                continue

            filtered_nodes.setdefault((source_type, source_id), source)
            filtered_nodes.setdefault((target_type, target_id), target)
            relationship_key = (
                source_type,
                source_id,
                relationship_type,
                target_type,
                target_id,
            )
            filtered_relationships[relationship_key] = relationship

        graph_document.nodes = list(filtered_nodes.values())
        graph_document.relationships = list(filtered_relationships.values())
        filtered_documents.append(graph_document)

    if dropped_nodes or dropped_relationships:
        logging.info(
            "Schema filter dropped %s nodes and %s relationships that were outside the allowed schema.",
            dropped_nodes,
            dropped_relationships,
        )

    return filtered_documents


def get_property_map(properties) -> dict:
    if isinstance(properties, dict):
        return properties
    return {}


def get_description(properties) -> str:
    description = get_property_map(properties).get("description")
    return str(description).strip() if description else ""


def set_description(properties, description: str) -> dict:
    property_map = dict(get_property_map(properties))
    if description:
        property_map["description"] = description
    return property_map


def get_strength(properties):
    return get_property_map(properties).get("strength")


def normalize_strength(value) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def set_strength(properties, strength: Optional[int]) -> dict:
    property_map = dict(get_property_map(properties))
    if strength is not None:
        property_map["strength"] = strength
    return property_map


def format_descriptions_for_prompt(descriptions: List[str]) -> str:
    return "\n".join(f"- {description}" for description in descriptions)


def build_description_summary_chain(llm, prompt_template: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DESCRIPTION_SUMMARIZATION_SYSTEM_PROMPT),
            ("human", prompt_template),
        ]
    )
    return prompt | llm | StrOutputParser()


def unique_descriptions(descriptions: List[str]) -> List[str]:
    unique_items: List[str] = []
    seen = set()
    for description in descriptions:
        normalized = normalize_graph_key(description)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_items.append(description.strip())
    return unique_items


def summarize_text_list(chain, payload: dict, descriptions: List[str]) -> str:
    unique_items = unique_descriptions(descriptions)
    if not unique_items:
        return ""
    if len(unique_items) == 1:
        return unique_items[0]

    try:
        summary = chain.invoke(
            {
                **payload,
                "descriptions": format_descriptions_for_prompt(unique_items),
            }
        ).strip()
        return summary or max(unique_items, key=len)
    except Exception:
        logging.warning("Description summarization failed; falling back to longest description.", exc_info=True)
        return max(unique_items, key=len)


def summarize_node_group(chain, node_key, nodes: List[Node], schema_profile: Optional[str]) -> Tuple[Tuple[str, str], str]:
    node_type, node_id = node_key
    descriptions = [get_description(node.properties) for node in nodes if get_description(node.properties)]
    summary = summarize_text_list(
        chain,
        {
            "entity_id": nodes[0].id,
            "entity_type": nodes[0].type,
            "schema_profile": schema_profile or "default",
        },
        descriptions,
    )
    return (node_type, node_id), summary


def summarize_relationship_group(
    chain,
    relationship_key,
    relationships: List[Relationship],
    schema_profile: Optional[str],
) -> Tuple[Tuple[str, str, str, str, str], str, Optional[int]]:
    relationship = relationships[0]
    descriptions = [get_description(rel.properties) for rel in relationships if get_description(rel.properties)]
    strengths = [
        normalize_strength(get_strength(rel.properties))
        for rel in relationships
        if normalize_strength(get_strength(rel.properties)) is not None
    ]
    summary = summarize_text_list(
        chain,
        {
            "source_id": relationship.source.id,
            "source_type": relationship.source.type,
            "relationship_type": relationship.type,
            "target_id": relationship.target.id,
            "target_type": relationship.target.type,
            "schema_profile": schema_profile or "default",
        },
        descriptions,
    )
    merged_strength = max(strengths) if strengths else None
    return relationship_key, summary, merged_strength


def summarize_graph_element_descriptions(
    graph_document_list: List[GraphDocument],
    llm,
    schema_profile: Optional[str] = None,
) -> List[GraphDocument]:
    node_groups: Dict[Tuple[str, str], List[Node]] = {}
    relationship_groups: Dict[Tuple[str, str, str, str, str], List[Relationship]] = {}

    for graph_document in graph_document_list:
        for node in graph_document.nodes:
            node_key = (normalize_graph_key(node.type), normalize_graph_key(node.id))
            node_groups.setdefault(node_key, []).append(node)
        for relationship in graph_document.relationships:
            relationship_key = (
                normalize_graph_key(relationship.source.type),
                normalize_graph_key(relationship.source.id),
                normalize_graph_key(relationship.type),
                normalize_graph_key(relationship.target.type),
                normalize_graph_key(relationship.target.id),
            )
            relationship_groups.setdefault(relationship_key, []).append(relationship)

    if not node_groups and not relationship_groups:
        return graph_document_list

    node_chain = build_description_summary_chain(llm, ENTITY_DESCRIPTION_SUMMARIZATION_PROMPT)
    relationship_chain = build_description_summary_chain(llm, RELATIONSHIP_DESCRIPTION_SUMMARIZATION_PROMPT)
    max_workers = max(1, get_value_from_env("DESCRIPTION_SUMMARIZATION_MAX_WORKERS", 4, int))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        node_futures = {
            executor.submit(summarize_node_group, node_chain, node_key, nodes, schema_profile): node_key
            for node_key, nodes in node_groups.items()
            if sum(1 for node in nodes if get_description(node.properties)) > 1
        }
        relationship_futures = {
            executor.submit(
                summarize_relationship_group,
                relationship_chain,
                relationship_key,
                relationships,
                schema_profile,
            ): relationship_key
            for relationship_key, relationships in relationship_groups.items()
            if sum(1 for relationship in relationships if get_description(relationship.properties)) > 1
        }

        for future in as_completed(node_futures):
            node_key, summary = future.result()
            if not summary:
                continue
            for node in node_groups[node_key]:
                node.properties = set_description(node.properties, summary)

        for future in as_completed(relationship_futures):
            relationship_key, summary, strength = future.result()
            for relationship in relationship_groups[relationship_key]:
                if summary:
                    relationship.properties = set_description(relationship.properties, summary)
                if strength is not None:
                    relationship.properties = set_strength(relationship.properties, strength)

    return graph_document_list


def build_second_pass_instructions(additional_instructions: str | None, schema_profile: Optional[str] = None) -> str:
    instructions = [SECOND_PASS_ADDITIONAL_INSTRUCTIONS]
    if schema_profile == EDUCATION_SCHEMA_PROFILE:
        instructions.append(EDUCATION_SECOND_PASS_ADDITIONAL_INSTRUCTIONS)
    if additional_instructions:
        instructions.append(additional_instructions)
    return "\n".join(instructions)
      

async def get_graph_document_list(
    llm, combined_chunk_document_list, allowedNodes, allowedRelationship,callback_handler, additional_instructions=None
):
    if additional_instructions:
        additional_instructions = sanitize_additional_instruction(additional_instructions)
    graph_document_list = []
    token_usage = 0
    try:
        if "diffbot_api_key" in dir(llm):
            llm_transformer = llm
            use_structured_output = False
        else:
            llm_model_name = get_llm_model_name(llm)
            force_disable_structured_output_models = {
                model_name.strip().lower()
                for model_name in get_value_from_env(
                    "DISABLE_STRUCTURED_OUTPUT_MODELS",
                    "qwen3.5-397b",
                    str,
                ).split(",")
                if model_name.strip()
            }
            force_disable_structured_output = llm_model_name in force_disable_structured_output_models

            if force_disable_structured_output:
                logging.info(
                    "Structured output disabled for model '%s'; using plain-text graph extraction mode",
                    llm_model_name,
                )
                supports_structured_output = False
            else:
                try:
                    llm.with_structured_output(_Graph)
                    supports_structured_output = True
                except Exception:
                    supports_structured_output = False
            use_structured_output = supports_structured_output and not isinstance(llm, ChatGroq)
            if use_structured_output:
                logging.info("LLM supports structured output; including descriptions and relationship strength in graph")
            else:
                logging.info("LLM does not support structured output; excluding rich graph properties")

            llm_transformer = build_graph_transformer(
                llm,
                allowedNodes,
                allowedRelationship,
                additional_instructions,
                use_structured_output,
            )

        try:
            if isinstance(llm,DiffbotGraphTransformer):
                graph_document_list = llm_transformer.convert_to_graph_documents(combined_chunk_document_list)
            else:
                graph_document_list = await llm_transformer.aconvert_to_graph_documents(combined_chunk_document_list)
        except Exception as error:
            if not isinstance(llm, DiffbotGraphTransformer) and use_structured_output and is_structured_output_validation_error(error):
                logging.warning(
                    "Structured output parsing failed for model '%s'; retrying graph extraction without structured output",
                    get_llm_model_name(llm),
                    exc_info=True,
                )
                llm_transformer = build_graph_transformer(
                    llm,
                    allowedNodes,
                    allowedRelationship,
                    additional_instructions,
                    False,
                )
                graph_document_list = await llm_transformer.aconvert_to_graph_documents(combined_chunk_document_list)
            else:
                raise
    except Exception as e:
       logging.error(f"Error in graph transformation: {e}", exc_info=True)
       raise LLMGraphBuilderException(f"Graph transformation failed: {str(e)}")
    finally:
        try:
            if callback_handler:
                usage = callback_handler.report()
                token_usage = usage.get("total_tokens", 0)
        except Exception as usage_err:
            logging.error(f"Error while reporting token usage: {usage_err}")

    return graph_document_list, token_usage

async def get_graph_from_llm(
    model,
    chunkId_chunkDoc_list,
    allowedNodes,
    allowedRelationship,
    chunks_to_combine,
    additional_instructions=None,
    schema_profile: Optional[str] = None,
):
   try:
       llm, model_name,callback_handler = get_llm(model)
       logging.info(f"Using model: {model_name}")
    
       combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list, chunks_to_combine)
       logging.info(f"Combined {len(combined_chunk_document_list)} chunks")

       (
           allowedNodes,
           allowedRelationship,
           additional_instructions,
           resolved_schema_profile,
       ) = resolve_schema_profile(
           allowedNodes,
           allowedRelationship,
           additional_instructions,
           schema_profile,
       )

       allowed_nodes = parse_allowed_nodes(allowedNodes)
       logging.info(f"Allowed nodes: {allowed_nodes}")

       allowed_relationships = parse_allowed_relationships(allowedRelationship, allowed_nodes)
       if allowed_relationships:
           logging.info(f"Allowed relationships: {allowed_relationships}")
       else:
           logging.info("No allowed relationships provided")

       graph_document_list,token_usage = await get_graph_document_list(
           llm,
           combined_chunk_document_list,
           allowed_nodes,
           allowed_relationships,
           callback_handler,
           additional_instructions,
       )
       graph_document_list = filter_graph_documents_by_schema(
           graph_document_list,
           allowed_nodes,
           allowed_relationships,
       )
       if get_value_from_env("ENABLE_GRAPH_EXTRACTION_SECOND_PASS", True, bool) and not isinstance(llm, DiffbotGraphTransformer):
           first_pass_counts = count_graph_elements(graph_document_list)
           logging.info(
               "Running second recall-focused graph extraction pass. First pass found %s nodes and %s relationships.",
               first_pass_counts[0],
               first_pass_counts[1],
           )
           try:
               second_pass_graph_documents, token_usage = await get_graph_document_list(
                   llm,
                   combined_chunk_document_list,
                   allowed_nodes,
                   allowed_relationships,
                   callback_handler,
                   build_second_pass_instructions(additional_instructions, resolved_schema_profile),
               )
               second_pass_graph_documents = filter_graph_documents_by_schema(
                   second_pass_graph_documents,
                   allowed_nodes,
                   allowed_relationships,
               )
               graph_document_list = merge_graph_document_lists(graph_document_list, second_pass_graph_documents)
               merged_counts = count_graph_elements(graph_document_list)
               logging.info(
                   "Second pass merged successfully. Graph now has %s nodes and %s relationships.",
                   merged_counts[0],
                   merged_counts[1],
               )
           except Exception:
               logging.warning("Second graph extraction pass failed; continuing with first-pass results.", exc_info=True)
       if get_value_from_env("ENABLE_DESCRIPTION_SUMMARIZATION", True, bool) and not isinstance(llm, DiffbotGraphTransformer):
           graph_document_list = summarize_graph_element_descriptions(
               graph_document_list,
               llm,
               resolved_schema_profile,
           )
           if callback_handler:
               usage = callback_handler.report()
               token_usage = usage.get("total_tokens", token_usage)
       logging.info(f"Generated {len(graph_document_list)} graph documents")
       return graph_document_list, token_usage
   except Exception as e:
       logging.error(f"Error in get_graph_from_llm: {e}", exc_info=True)
       raise LLMGraphBuilderException(f"Error in getting graph from llm: {e}")

def sanitize_additional_instruction(instruction: str) -> str:
   """
   Sanitizes additional instruction by:
   - Replacing curly braces `{}` with `[]` to prevent variable interpretation.
   - Removing potential injection patterns like `os.getenv()`, `eval()`, `exec()`.
   - Stripping problematic special characters.
   - Normalizing whitespace.
   Args:
       instruction (str): Raw additional instruction input.
   Returns:
       str: Sanitized instruction safe for LLM processing.
   """
   logging.info("Sanitizing additional instructions")
   instruction = instruction.replace("{", "[").replace("}", "]")  # Convert `{}` to `[]` for safety
   # Step 2: Block dangerous function calls
   injection_patterns = [r"os\.getenv\(", r"eval\(", r"exec\(", r"subprocess\.", r"import os", r"import subprocess"]
   for pattern in injection_patterns:
       instruction = re.sub(pattern, "[BLOCKED]", instruction, flags=re.IGNORECASE)
   # Step 4: Normalize spaces
   instruction = re.sub(r'\s+', ' ', instruction).strip()
   return instruction
