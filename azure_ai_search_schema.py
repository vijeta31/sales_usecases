# Azure AI Search Content Storage with Metadata Implementation
# For Pain Point Analysis RAG System

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch
)
from azure.core.credentials import AzureKeyCredential
from azure.ai.openai import AzureOpenAIClient
from datetime import datetime
import json
from typing import List, Dict, Any
import hashlib

# Configuration
SEARCH_ENDPOINT = "https://your-search-service.search.windows.net"
SEARCH_API_KEY = "your-api-key"
INDEX_NAME = "pain-points-rag-index"
OPENAI_ENDPOINT = "https://your-openai.openai.azure.com"
OPENAI_API_KEY = "your-openai-key"
EMBEDDING_MODEL = "text-embedding-3-large"

class PainPointSearchIndexManager:
    """Manages Azure AI Search index creation and document storage with metadata"""
    
    def __init__(self):
        self.credential = AzureKeyCredential(SEARCH_API_KEY)
        self.index_client = SearchIndexClient(
            endpoint=SEARCH_ENDPOINT,
            credential=self.credential
        )
        self.search_client = None
        
    def create_index_schema(self):
        """Create index schema with comprehensive metadata fields for pain point analysis"""
        
        # Define the fields for the index
        fields = [
            # Document ID (required, key field)
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                sortable=True,
                filterable=True
            ),
            
            # Content fields
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True,
                analyzer_name="en.microsoft"  # Or "en.lucene" for more control
            ),
            
            # Chunk information
            SimpleField(
                name="chunk_id",
                type=SearchFieldDataType.String,
                filterable=True,
                sortable=True
            ),
            
            SimpleField(
                name="chunk_index",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True
            ),
            
            SimpleField(
                name="parent_document_id",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            
            # Vector embedding field
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # For text-embedding-3-large
                vector_search_profile_name="vector-profile"
            ),
            
            # Customer metadata
            SimpleField(
                name="customer_id",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            
            SimpleField(
                name="customer_name",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            
            SimpleField(
                name="customer_tier",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True  # Enterprise, Standard, Basic
            ),
            
            SimpleField(
                name="customer_region",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            
            # Communication metadata
            SimpleField(
                name="communication_type",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True  # email, ticket, note, chat
            ),
            
            SimpleField(
                name="communication_date",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True
            ),
            
            SearchableField(
                name="subject",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True
            ),
            
            SimpleField(
                name="priority",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True  # high, medium, low
            ),
            
            # Product metadata
            SimpleField(
                name="product_id",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            
            SimpleField(
                name="product_name",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            
            SimpleField(
                name="product_category",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            
            # Issue metadata
            SimpleField(
                name="issue_type",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True  # billing, technical, service, access
            ),
            
            SimpleField(
                name="issue_category",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            
            # Sentiment and analysis metadata
            SimpleField(
                name="sentiment_score",
                type=SearchFieldDataType.Double,
                filterable=True,
                sortable=True  # -1.0 to 1.0
            ),
            
            SimpleField(
                name="sentiment_label",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True  # positive, negative, neutral
            ),
            
            SimpleField(
                name="urgency_score",
                type=SearchFieldDataType.Double,
                filterable=True,
                sortable=True  # 0.0 to 1.0
            ),
            
            # Extracted pain points (stored as JSON string)
            SearchableField(
                name="pain_points",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True
            ),
            
            # Tags for flexible categorization
            SearchableField(
                name="tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                searchable=True,
                filterable=True,
                facetable=True
            ),
            
            # Processing metadata
            SimpleField(
                name="processed_date",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True
            ),
            
            SimpleField(
                name="processing_version",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            
            # Additional metadata as JSON
            SimpleField(
                name="metadata_json",
                type=SearchFieldDataType.String,
                retrievable=True
            )
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-config",
                    parameters={
                        "m": 4,  # Number of bi-directional links
                        "efConstruction": 400,  # Size of dynamic list
                        "efSearch": 500,  # Size of dynamic list for search
                        "metric": "cosine"
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-config"
                )
            ]
        )
        
        # Configure semantic search for better relevance
        semantic_search = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="semantic-config",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="subject"),
                        content_fields=[
                            SemanticField(field_name="content"),
                            SemanticField(field_name="pain_points")
                        ],
                        keywords_fields=[
                            SemanticField(field_name="tags")
                        ]
                    )
                )
            ]
        )
        
        # Create the index
        index = SearchIndex(
            name=INDEX_NAME,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )
        
        return index
    
    def create_or_update_index(self):
        """Create or update the search index"""
        index = self.create_index_schema()
        
        try:
            self.index_client.create_or_update_index(index)
            print(f"Index '{INDEX_NAME}' created/updated successfully")
            
            # Initialize search client
            self.search_client = SearchClient(
                endpoint=SEARCH_ENDPOINT,
                index_name=INDEX_NAME,
                credential=self.credential
            )
        except Exception as e:
            print(f"Error creating index: {e}")
            raise
    
    def prepare_document_for_indexing(
        self, 
        content: str,
        chunk_index: int,
        parent_doc_id: str,
        metadata: Dict[str, Any],
        embedding: List[float]
    ) -> Dict[str, Any]:
        """Prepare a document with all metadata for indexing"""
        
        # Generate unique ID for this chunk
        chunk_id = hashlib.md5(
            f"{parent_doc_id}_{chunk_index}_{content[:50]}".encode()
        ).hexdigest()
        
        # Extract pain points (this would be done by your LLM in practice)
        pain_points = metadata.get("extracted_pain_points", [])
        
        # Prepare the document
        document = {
            "id": chunk_id,
            "content": content,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "parent_document_id": parent_doc_id,
            "content_vector": embedding,
            
            # Customer metadata
            "customer_id": metadata.get("customer_id"),
            "customer_name": metadata.get("customer_name"),
            "customer_tier": metadata.get("customer_tier"),
            "customer_region": metadata.get("customer_region"),
            
            # Communication metadata
            "communication_type": metadata.get("communication_type"),
            "communication_date": metadata.get("communication_date"),
            "subject": metadata.get("subject", ""),
            "priority": metadata.get("priority", "medium"),
            
            # Product metadata
            "product_id": metadata.get("product_id"),
            "product_name": metadata.get("product_name"),
            "product_category": metadata.get("product_category"),
            
            # Issue metadata
            "issue_type": metadata.get("issue_type"),
            "issue_category": metadata.get("issue_category"),
            
            # Sentiment analysis
            "sentiment_score": metadata.get("sentiment_score", 0.0),
            "sentiment_label": metadata.get("sentiment_label", "neutral"),
            "urgency_score": metadata.get("urgency_score", 0.0),
            
            # Pain points as JSON string
            "pain_points": json.dumps(pain_points),
            
            # Tags
            "tags": metadata.get("tags", []),
            
            # Processing metadata
            "processed_date": datetime.utcnow().isoformat(),
            "processing_version": "1.0",
            
            # Additional metadata
            "metadata_json": json.dumps(metadata.get("additional_metadata", {}))
        }
        
        return document
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index multiple documents with metadata"""
        try:
            result = self.search_client.upload_documents(documents=documents)
            print(f"Indexed {len(documents)} documents")
            return result
        except Exception as e:
            print(f"Error indexing documents: {e}")
            raise


class PainPointRAGSearcher:
    """Handles searching and retrieval from the pain point index"""
    
    def __init__(self):
        self.credential = AzureKeyCredential(SEARCH_API_KEY)
        self.search_client = SearchClient(
            endpoint=SEARCH_ENDPOINT,
            index_name=INDEX_NAME,
            credential=self.credential
        )
    
    def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        filters: Dict[str, Any] = None,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search with metadata filtering"""
        
        # Build filter expression
        filter_expr = self._build_filter_expression(filters) if filters else None
        
        # Perform hybrid search
        results = self.search_client.search(
            search_text=query,
            vector_queries=[{
                "vector": query_vector,
                "k_nearest_neighbors": top_k,
                "fields": "content_vector"
            }],
            filter=filter_expr,
            select=[
                "id", "content", "customer_id", "customer_name", 
                "customer_tier", "product_name", "issue_type",
                "sentiment_label", "pain_points", "communication_date",
                "subject", "priority", "tags"
            ],
            top=top_k,
            query_type="semantic",
            semantic_configuration_name="semantic-config"
        )
        
        return [doc for doc in results]
    
    def _build_filter_expression(self, filters: Dict[str, Any]) -> str:
        """Build OData filter expression from filter dictionary"""
        filter_parts = []
        
        # Customer filters
        if "customer_id" in filters:
            filter_parts.append(f"customer_id eq '{filters['customer_id']}'")
        
        if "customer_tier" in filters:
            if isinstance(filters["customer_tier"], list):
                tier_filters = " or ".join([f"customer_tier eq '{tier}'" for tier in filters["customer_tier"]])
                filter_parts.append(f"({tier_filters})")
            else:
                filter_parts.append(f"customer_tier eq '{filters['customer_tier']}'")
        
        # Date range filter
        if "date_from" in filters:
            filter_parts.append(f"communication_date ge {filters['date_from'].isoformat()}")
        
        if "date_to" in filters:
            filter_parts.append(f"communication_date le {filters['date_to'].isoformat()}")
        
        # Product filter
        if "product_id" in filters:
            filter_parts.append(f"product_id eq '{filters['product_id']}'")
        
        # Issue type filter
        if "issue_type" in filters:
            filter_parts.append(f"issue_type eq '{filters['issue_type']}'")
        
        # Sentiment filter
        if "sentiment_label" in filters:
            filter_parts.append(f"sentiment_label eq '{filters['sentiment_label']}'")
        
        # Priority filter
        if "priority" in filters:
            filter_parts.append(f"priority eq '{filters['priority']}'")
        
        # Combine all filters with AND
        return " and ".join(filter_parts) if filter_parts else None
    
    def search_by_pain_point_pattern(
        self,
        pain_point_keywords: List[str],
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for specific pain point patterns"""
        
        # Build search query from pain point keywords
        search_query = " OR ".join([f'"{keyword}"' for keyword in pain_point_keywords])
        
        results = self.search_client.search(
            search_text=search_query,
            search_fields=["content", "pain_points", "tags"],
            filter=self._build_filter_expression(filters) if filters else None,
            select=[
                "id", "content", "customer_tier", "product_name",
                "pain_points", "sentiment_score", "urgency_score"
            ],
            top=50
        )
        
        return [doc for doc in results]


# Example usage
if __name__ == "__main__":
    # Initialize index manager
    index_manager = PainPointSearchIndexManager()
    
    # Create or update index
    index_manager.create_or_update_index()
    
    # Example document with comprehensive metadata
    example_metadata = {
        "customer_id": "CUST-12345",
        "customer_name": "Acme Corporation",
        "customer_tier": "enterprise",
        "customer_region": "north_america",
        "communication_type": "email",
        "communication_date": datetime(2025, 8, 4, 10, 30, 0),
        "subject": "Billing discrepancies and system access issues",
        "priority": "high",
        "product_id": "PROD-BILLING-001",
        "product_name": "Enterprise Billing System",
        "product_category": "billing",
        "issue_type": "billing",
        "issue_category": "incorrect_charges",
        "sentiment_score": -0.7,
        "sentiment_label": "negative",
        "urgency_score": 0.85,
        "extracted_pain_points": [
            {
                "description": "Incorrect monthly charges for the past 3 months",
                "severity": "high",
                "frequency": "recurring"
            },
            {
                "description": "Unable to access billing dashboard",
                "severity": "medium",
                "frequency": "persistent"
            }
        ],
        "tags": ["billing_error", "access_issue", "enterprise_customer", "high_priority"],
        "additional_metadata": {
            "ticket_id": "TKT-98765",
            "assigned_team": "billing_support",
            "sla_deadline": "2025-08-05T17:00:00Z"
        }
    }
    
    # Example content chunk
    content = """
    We have been experiencing significant billing discrepancies for the past three months. 
    Our monthly charges have been consistently higher than our contracted rate by approximately 
    $5,000. Additionally, our finance team is unable to access the billing dashboard to review 
    detailed invoices, which is critical for our month-end reconciliation process.
    """
    
    # Generate embedding (in practice, use Azure OpenAI)
    dummy_embedding = [0.1] * 1536  # Placeholder
    
    # Prepare document
    document = index_manager.prepare_document_for_indexing(
        content=content,
        chunk_index=0,
        parent_doc_id="DOC-2025-08-04-001",
        metadata=example_metadata,
        embedding=dummy_embedding
    )
    
    # Index the document
    index_manager.index_documents([document])
    
    # Example search with filters
    searcher = PainPointRAGSearcher()
    
    # Search for enterprise customers with billing issues
    filters = {
        "customer_tier": "enterprise",
        "issue_type": "billing",
        "date_from": datetime(2025, 7, 1),
        "sentiment_label": "negative"
    }
    
    # Perform search (would need actual query embedding)
    results = searcher.hybrid_search(
        query="billing discrepancies incorrect charges",
        query_vector=dummy_embedding,
        filters=filters,
        top_k=10
    )
    
    print(f"Found {len(results)} relevant documents")
