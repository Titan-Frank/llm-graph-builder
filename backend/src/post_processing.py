from neo4j import GraphDatabase
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_neo4j import Neo4jGraph
import os
from src.graph_query import get_graphDB_driver
from src.shared.common_fn import load_embedding_model,execute_graph_query,get_value_from_env
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.shared.constants import (
    EDUCATION_SCHEMA_PRESET,
    ENTITY_DESCRIPTION_SUMMARIZATION_PROMPT,
    GRAPH_CLEANUP_PROMPT,
    RELATIONSHIP_DESCRIPTION_SUMMARIZATION_PROMPT,
)
from src.llm import build_description_summary_chain, get_llm, summarize_text_list
from src.graphDB_dataAccess import graphDBdataAccess
import time 

# Constants for Full-Text Indexes
LABELS_QUERY = "CALL db.labels()"
FILTER_LABELS = ["Chunk","Document","__Community__"]
FULL_TEXT_QUERY = "CREATE FULLTEXT INDEX entities FOR (n{labels_str}) ON EACH [n.id, n.description];"
HYBRID_SEARCH_FULL_TEXT_QUERY = "CREATE FULLTEXT INDEX keyword FOR (n:Chunk) ON EACH [n.text]" 
COMMUNITY_INDEX_FULL_TEXT_QUERY = "CREATE FULLTEXT INDEX community_keyword FOR (n:`__Community__`) ON EACH [n.summary]" 

# Constants for Vector Indexes
CHUNK_VECTOR_INDEX_NAME = "vector"
ENTITY_VECTOR_INDEX_NAME = "entity_vector"
VECTOR_EMBEDDING_DEFAULT_DIMENSION = 384

CREATE_VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX {index_name} IF NOT EXISTS FOR (n:{node_label}) ON (n.{embedding_property})
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {embedding_dimension},
    `vector.similarity_function`: 'cosine'
  }}
}}
"""

# Index Configurations
FULLTEXT_INDEXES = [
    {"type": "entities", "query": FULL_TEXT_QUERY},
    {"type": "hybrid", "query": HYBRID_SEARCH_FULL_TEXT_QUERY},
    {"type": "community", "query": COMMUNITY_INDEX_FULL_TEXT_QUERY}
]

VECTOR_INDEXES = [
    {"name": CHUNK_VECTOR_INDEX_NAME, "label": "Chunk", "property": "embedding"},
    {"name": ENTITY_VECTOR_INDEX_NAME, "label": "__Entity__", "property": "embedding"}
]

def create_vector_index(session, index_name, node_label, embedding_property, embedding_dimension):
    """Creates a vector index in the Neo4j database."""
    drop_query = f"DROP INDEX {index_name} IF EXISTS;"
    session.run(drop_query)
    
    query = CREATE_VECTOR_INDEX_QUERY.format(
        index_name=index_name,
        node_label=node_label,
        embedding_property=embedding_property,
        embedding_dimension=embedding_dimension
    )
    session.run(query)
    logging.info(f"Vector index '{index_name}' created successfully.")

def create_fulltext_index(session, index_type, query):
    """Creates a full-text index in the Neo4j database."""
    drop_query = f"DROP INDEX {index_type} IF EXISTS;"
    if index_type == 'hybrid':
        drop_query = "DROP INDEX keyword IF EXISTS;"
    elif index_type == 'community':
        drop_query = "DROP INDEX community_keyword IF EXISTS;"
    
    session.run(drop_query)

    if index_type == "entities":
        result = session.run(LABELS_QUERY)
        labels = [record["label"] for record in result if record["label"] not in FILTER_LABELS]
        if labels:
            labels_str = ":" + "|".join([f"`{label}`" for label in labels])
            query = query.format(labels_str=labels_str)
        else:
            logging.info("Full-text index for entities not created as no labels were found.")
            return
            
    session.run(query)
    logging.info(f"Full-text index for '{index_type}' created successfully.")


def create_vector_fulltext_indexes(credentials, embedding_provider, embedding_model):
    """Creates all configured full-text and vector indexes."""
    logging.info("Starting the process of creating full-text and vector indexes.")
    
    _, dimension = load_embedding_model(embedding_provider, embedding_model)
    if not dimension:
        dimension = VECTOR_EMBEDDING_DEFAULT_DIMENSION

    try:
        driver = get_graphDB_driver(credentials)
        driver.verify_connectivity()
        logging.info("Database connectivity verified.")

        with driver.session() as session:
            # Create Full-Text Indexes
            for index_config in FULLTEXT_INDEXES:
                try:
                    create_fulltext_index(session, index_config["type"], index_config["query"])
                except Exception as e:
                    logging.error(f"Failed to create full-text index for type '{index_config['type']}': {e}")

            # Create Vector Indexes
            for index_config in VECTOR_INDEXES:
                try:
                    create_vector_index(session, index_config["name"], index_config["label"], index_config["property"], dimension)
                except Exception as e:
                    logging.error(f"Failed to create vector index '{index_config['name']}': {e}")

    except Exception as e:
        logging.error(f"An error occurred during the index creation process: {e}", exc_info=True)
    finally:
        if 'driver' in locals() and driver:
            driver.close()
            logging.info("Driver closed successfully.")
    
    logging.info("Full-text and vector index creation process completed.")


def create_entity_embedding(graph:Neo4jGraph, embedding_provider, embedding_model):
    rows = fetch_entities_for_embedding(graph)
    for i in range(0, len(rows), 1000):
        update_embeddings(rows[i:i+1000],graph, embedding_provider, embedding_model)
            
def fetch_entities_for_embedding(graph):
    query = """
                MATCH (e)
                WHERE NOT (e:Chunk OR e:Document OR e:`__Community__`) AND e.embedding IS NULL AND e.id IS NOT NULL
                RETURN elementId(e) AS elementId, e.id + " " + coalesce(e.description, "") AS text
                """ 
    result = execute_graph_query(graph,query)        
    return [{"elementId": record["elementId"], "text": record["text"]} for record in result]

def update_embeddings(rows, graph, embedding_provider, embedding_model):
    embeddings, dimension = load_embedding_model(embedding_provider, embedding_model)
    logging.info(f"update embedding for entities")
    for row in rows:
        row['embedding'] = embeddings.embed_query(row['text'])                        
    query = """
      UNWIND $rows AS row
      MATCH (e) WHERE elementId(e) = row.elementId
      CALL db.create.setNodeVectorProperty(e, "embedding", row.embedding)
      """  
    return execute_graph_query(graph,query,params={'rows':rows})          

def graph_schema_consolidation(graph):
    graphDb_data_Access = graphDBdataAccess(graph)
    node_labels,relation_labels = graphDb_data_Access.get_nodelabels_relationships()
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate(
        messages=[("system", GRAPH_CLEANUP_PROMPT), ("human", "{input}")],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    graph_cleanup_model = get_value_from_env("GRAPH_CLEANUP_MODEL", 'openai_gpt_5_mini')
    llm, _, _ = get_llm(graph_cleanup_model)
    chain = prompt | llm | parser

    nodes_relations_input = {'nodes': node_labels, 'relationships': relation_labels}
    mappings = chain.invoke({'input': nodes_relations_input})
    node_mapping = {old: new for new, old_list in mappings['nodes'].items() for old in old_list if new != old}
    relation_mapping = {old: new for new, old_list in mappings['relationships'].items() for old in old_list if new != old}

    logging.info(f"Node Labels: Total = {len(node_labels)}, Reduced to = {len(set(node_mapping.values()))} (from {len(node_mapping)})")
    logging.info(f"Relationship Types: Total = {len(relation_labels)}, Reduced to = {len(set(relation_mapping.values()))} (from {len(relation_mapping)})")

    if node_mapping:
        for old_label, new_label in node_mapping.items():
            query = f"""
                    MATCH (n:`{old_label}`)
                    SET n:`{new_label}`
                    REMOVE n:`{old_label}`
                    """
            execute_graph_query(graph,query)

    for old_label, new_label in relation_mapping.items():
        query = f"""
                MATCH (n)-[r:`{old_label}`]->(m)
                CREATE (n)-[r2:`{new_label}`]->(m)
                DELETE r
                """
        execute_graph_query(graph,query)

    return None


def deduplicate_parallel_relationships(graph):
    query = """
    MATCH (source:__Entity__)-[r]->(target:__Entity__)
    WHERE NOT type(r) IN ['HAS_ENTITY', 'PART_OF', 'NEXT_CHUNK', 'SIMILAR', 'IN_COMMUNITY', 'PARENT_COMMUNITY', 'FIRST_CHUNK']
    WITH source, target, type(r) AS relationship_type, collect(r) AS relationships
    WHERE size(relationships) > 1
    WITH relationships,
         reduce(
           description_candidates = [],
           rel IN relationships |
           description_candidates +
           coalesce(rel.description_candidates, []) +
           CASE
             WHEN rel.description IS NULL OR trim(toString(rel.description)) = '' THEN []
             ELSE [toString(rel.description)]
           END
         ) AS merged_description_candidates,
         reduce(
           strength_candidates = [],
           rel IN relationships |
           strength_candidates +
           coalesce(rel.strength_candidates, []) +
           CASE
             WHEN rel.strength IS NULL OR trim(toString(rel.strength)) = '' THEN []
             ELSE [toInteger(rel.strength)]
           END
         ) AS merged_strength_candidates
    WITH head(relationships) AS primary_relationship,
         tail(relationships) AS duplicate_relationships,
         apoc.coll.toSet([candidate IN merged_description_candidates WHERE candidate IS NOT NULL AND trim(toString(candidate)) <> '']) AS deduped_description_candidates,
         apoc.coll.toSet([candidate IN merged_strength_candidates WHERE candidate IS NOT NULL]) AS deduped_strength_candidates
    SET primary_relationship.description_candidates =
          CASE
            WHEN size(deduped_description_candidates) = 0 THEN coalesce(primary_relationship.description_candidates, [])
            ELSE deduped_description_candidates
          END,
        primary_relationship.strength_candidates =
          CASE
            WHEN size(deduped_strength_candidates) = 0 THEN coalesce(primary_relationship.strength_candidates, [])
            ELSE deduped_strength_candidates
          END,
        primary_relationship.strength =
          CASE
            WHEN size(deduped_strength_candidates) = 0 THEN primary_relationship.strength
            ELSE reduce(max_strength = null, candidate IN deduped_strength_candidates |
              CASE
                WHEN max_strength IS NULL OR candidate > max_strength THEN candidate
                ELSE max_strength
              END
            )
          END
    FOREACH (relationship IN duplicate_relationships | DELETE relationship)
    RETURN count(primary_relationship) AS deduplicated_relationship_groups,
           sum(size(duplicate_relationships)) AS deleted_relationships
    """
    result = execute_graph_query(graph, query)
    if result:
        logging.info(
            "Deduplicated %s relationship groups and deleted %s duplicate relationships.",
            result[0].get("deduplicated_relationship_groups", 0),
            result[0].get("deleted_relationships", 0),
        )
    else:
        logging.info("No duplicate relationships found to deduplicate.")
    return result


def infer_graph_schema_profile(graph) -> str | None:
    graphDb_data_Access = graphDBdataAccess(graph)
    node_labels, _ = graphDb_data_Access.get_nodelabels_relationships()
    normalized_labels = {label.strip().lower() for label in node_labels}
    education_labels = {label.strip().lower() for label in EDUCATION_SCHEMA_PRESET["allowed_nodes"]}
    if len(normalized_labels & education_labels) >= 3:
        return EDUCATION_SCHEMA_PRESET["name"]
    return None


def fetch_entity_description_candidates(graph):
    query = """
    MATCH (n:__Entity__)
    WHERE n.id IS NOT NULL
    WITH n,
         [candidate IN coalesce(n.description_candidates, []) WHERE candidate IS NOT NULL AND trim(toString(candidate)) <> ''] AS candidates
    WITH n,
         CASE
           WHEN size(candidates) > 0 THEN candidates
           WHEN n.description IS NOT NULL AND trim(toString(n.description)) <> '' THEN [toString(n.description)]
           ELSE []
         END AS descriptions
    WHERE size(descriptions) > 0
    RETURN
      elementId(n) AS element_id,
      n.id AS entity_id,
      head([label IN labels(n) WHERE label <> '__Entity__']) AS entity_type,
      descriptions
    """
    return execute_graph_query(graph, query)


def fetch_relationship_description_candidates(graph):
    query = """
    MATCH (source:__Entity__)-[r]->(target:__Entity__)
    WHERE NOT type(r) IN ['HAS_ENTITY', 'PART_OF', 'NEXT_CHUNK', 'SIMILAR', 'IN_COMMUNITY', 'PARENT_COMMUNITY', 'FIRST_CHUNK']
    WITH source, r, target,
         [candidate IN coalesce(r.description_candidates, []) WHERE candidate IS NOT NULL AND trim(toString(candidate)) <> ''] AS description_candidates,
         [candidate IN coalesce(r.strength_candidates, []) WHERE candidate IS NOT NULL] AS strength_candidates
    WITH source, r, target,
         CASE
           WHEN size(description_candidates) > 0 THEN description_candidates
           WHEN r.description IS NOT NULL AND trim(toString(r.description)) <> '' THEN [toString(r.description)]
           ELSE []
         END AS descriptions,
         CASE
           WHEN size(strength_candidates) > 0 THEN strength_candidates
           WHEN r.strength IS NOT NULL THEN [toInteger(r.strength)]
           ELSE []
         END AS strengths
    WHERE size(descriptions) > 0 OR size(strengths) > 0
    RETURN
      elementId(r) AS element_id,
      source.id AS source_id,
      head([label IN labels(source) WHERE label <> '__Entity__']) AS source_type,
      type(r) AS relationship_type,
      target.id AS target_id,
      head([label IN labels(target) WHERE label <> '__Entity__']) AS target_type,
      descriptions,
      strengths
    """
    return execute_graph_query(graph, query)


def summarize_entity_candidate_row(chain, row, schema_profile):
    description = summarize_text_list(
        chain,
        {
            "entity_id": row["entity_id"],
            "entity_type": row["entity_type"] or "__Entity__",
            "schema_profile": schema_profile or "default",
        },
        row["descriptions"],
    )
    return {"element_id": row["element_id"], "description": description}


def summarize_relationship_candidate_row(chain, row, schema_profile):
    description = summarize_text_list(
        chain,
        {
            "source_id": row["source_id"],
            "source_type": row["source_type"] or "__Entity__",
            "relationship_type": row["relationship_type"],
            "target_id": row["target_id"],
            "target_type": row["target_type"] or "__Entity__",
            "schema_profile": schema_profile or "default",
        },
        row["descriptions"],
    )
    strengths = [int(value) for value in row.get("strengths", []) if value is not None]
    return {
        "element_id": row["element_id"],
        "description": description,
        "strength": max(strengths) if strengths else None,
    }


def write_entity_description_summaries(graph, rows):
    if not rows:
        return
    query = """
    UNWIND $rows AS row
    MATCH (n) WHERE elementId(n) = row.element_id
    SET n.description = row.description
    """
    execute_graph_query(graph, query, params={"rows": rows})


def write_relationship_description_summaries(graph, rows):
    if not rows:
        return
    query = """
    UNWIND $rows AS row
    MATCH ()-[r]->() WHERE elementId(r) = row.element_id
    SET r.description = row.description,
        r.strength = CASE WHEN row.strength IS NULL THEN r.strength ELSE row.strength END
    """
    execute_graph_query(graph, query, params={"rows": rows})


def consolidate_graph_element_descriptions(graph, model=None):
    schema_profile = infer_graph_schema_profile(graph)
    model_name = model or get_value_from_env("DESCRIPTION_SUMMARIZATION_MODEL", "openai_gpt_5_mini")
    llm, _, _ = get_llm(model_name)

    entity_rows = fetch_entity_description_candidates(graph)
    relationship_rows = fetch_relationship_description_candidates(graph)

    entity_chain = build_description_summary_chain(llm, ENTITY_DESCRIPTION_SUMMARIZATION_PROMPT)
    relationship_chain = build_description_summary_chain(llm, RELATIONSHIP_DESCRIPTION_SUMMARIZATION_PROMPT)
    max_workers = max(1, get_value_from_env("DESCRIPTION_SUMMARIZATION_MAX_WORKERS", 4, int))

    summarized_entities = []
    summarized_relationships = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        entity_futures = [
            executor.submit(summarize_entity_candidate_row, entity_chain, row, schema_profile)
            for row in entity_rows
        ]
        relationship_futures = [
            executor.submit(summarize_relationship_candidate_row, relationship_chain, row, schema_profile)
            for row in relationship_rows
        ]

        for future in as_completed(entity_futures):
            summarized_entities.append(future.result())

        for future in as_completed(relationship_futures):
            summarized_relationships.append(future.result())

    write_entity_description_summaries(graph, summarized_entities)
    write_relationship_description_summaries(graph, summarized_relationships)
    logging.info(
        "Consolidated descriptions for %s entities and %s relationships.",
        len(summarized_entities),
        len(summarized_relationships),
    )
