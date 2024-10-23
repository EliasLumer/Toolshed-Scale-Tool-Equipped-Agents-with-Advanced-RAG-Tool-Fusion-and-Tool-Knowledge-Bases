from abc import ABC, abstractmethod
from langchain.schema.document import Document
import re
from typing import Dict, Any, List, Optional

class BaseKnowledgeBaseBuilder(ABC):
    def __init__(self, toolshed_dict: Dict[str, Any]):
        self.toolshed_dict = toolshed_dict

    @abstractmethod
    def build_documents(self, *args, **kwargs) -> List[Document]:
        pass

class DocumentBuilder:
    def __init__(self, toolshed_dict: Dict[str, Any]):
        self.toolshed_dict = toolshed_dict

    def _get_args_schema(self, tool_name: str) -> str:
        tool_object = self.toolshed_dict[tool_name]['tool_object']
        try:
            schema = tool_object.args_schema.schema()
            parameters = []
            for key, value in schema['properties'].items():
                parameters.append(f"{value['title']}: {value['description']}")
            return ' '.join(parameters)
        except AttributeError:
            return ''

    def _get_hypothetical_questions(self, tool_name: str, hypothetical_questions_dict: Dict[str, List[str]]) -> str:
        questions = hypothetical_questions_dict.get(tool_name, [])
        return ' '.join(questions)

    def _get_key_topics(self, tool_name: str, key_topics_dict: Dict[str, List[str]]) -> str:
        topics = key_topics_dict.get(tool_name, [])
        return ' '.join(topics)

    def _format_tool_name_for_embedding(self, tool_name: str) -> str:
        """
        Formats the tool name for embedding by:
        - Replacing underscores with spaces.
        - Inserting spaces before uppercase letters (for CamelCase and mixedCase).
        - Converting the result to title case.
        """
        # Replace underscores with spaces
        name_with_spaces = tool_name.replace('_', ' ')
        # Insert spaces before uppercase letters (excluding the first character)
        name_with_spaces = re.sub(r'(?<!^)(?=[A-Z])', ' ', name_with_spaces)
        # Convert to title case
        tool_name_for_embedding = name_with_spaces.title()
        return tool_name_for_embedding

    def build_document(
        self,
        tool_name: str,
        include_name: bool = True,
        include_description: bool = True,
        include_args_schema: bool = False,
        include_hypothetical_questions: bool = False,
        include_key_topics: bool = False,
        hypothetical_questions_dict: Optional[Dict[str, List[str]]] = None,
        key_topics_dict: Optional[Dict[str, List[str]]] = None,
    ) -> Document:
        # Build the document content based on the specified components
        content_parts = []

        tool_data_dict = self.toolshed_dict[tool_name]
        tool_description = tool_data_dict['tool_object'].description

        if include_name:
            tool_name_for_embedding = self._format_tool_name_for_embedding(tool_name)
            content_parts.append(tool_name_for_embedding)
        if include_description:
            content_parts.append(tool_description)
        if include_args_schema:
            args_schema = self._get_args_schema(tool_name)
            if args_schema:
                content_parts.append(args_schema)
        if include_hypothetical_questions:
            if hypothetical_questions_dict is None:
                raise ValueError("hypothetical_questions_dict must be provided when include_hypothetical_questions is True")
            questions_text = self._get_hypothetical_questions(tool_name, hypothetical_questions_dict)
            if questions_text:
                content_parts.append(questions_text)
        if include_key_topics:
            if key_topics_dict is None:
                raise ValueError("key_topics_dict must be provided when include_key_topics is True")
            key_topics_text = self._get_key_topics(tool_name, key_topics_dict)
            if key_topics_text:
                content_parts.append(key_topics_text)

        page_content = ' - '.join(content_parts)

        metadata = {
            'tool_name': tool_name,
            'tool_hash':'Not implemented yet',
            # Add any other metadata you need
        }

        doc = Document(page_content=page_content, metadata=metadata)
        return doc

class ToolshedKnowledgeBaseBuilder(BaseKnowledgeBaseBuilder):
    def __init__(self, toolshed_dict: Dict[str, Any]):
        super().__init__(toolshed_dict)
        self.document_builder = DocumentBuilder(toolshed_dict)

    def build_documents(
        self,
        tool_names: Optional[List[str]] = None,
        include_name: bool = True,
        include_description: bool = True,
        include_args_schema: bool = False,
        include_hypothetical_questions: bool = False,
        include_key_topics: bool = False,
        hypothetical_questions_dict: Optional[Dict[str, List[str]]] = None,
        key_topics_dict: Optional[Dict[str, List[str]]] = None,
    ) -> List[Document]:
        if not tool_names:
            tool_names = list(self.toolshed_dict.keys())

        docs = []
        for tool_name in tool_names:
            doc = self.document_builder.build_document(
                tool_name=tool_name,
                include_name=include_name,
                include_description=include_description,
                include_args_schema=include_args_schema,
                include_hypothetical_questions=include_hypothetical_questions,
                include_key_topics=include_key_topics,
                hypothetical_questions_dict=hypothetical_questions_dict,
                key_topics_dict=key_topics_dict,
            )
            docs.append(doc)
        return docs
