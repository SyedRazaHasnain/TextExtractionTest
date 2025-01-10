"""OpenAI LLM client for regulatory text analysis."""

import json
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential

from openai import AsyncOpenAI, APIError, RateLimitError
from openai.types.chat import ChatCompletion

from .config import OpenAIConfig

logger = logging.getLogger(__name__)

@dataclass
class AnalysisPrompt:
    """Structure for analysis prompts."""
    system_prompt: str
    user_prompt_template: str

class LLMClient:
    """Client for interacting with OpenAI's LLM."""

    # Default prompts for different analysis types
    DEFAULT_PROMPTS = {
        'regulatory': AnalysisPrompt(
            system_prompt="""You are an expert regulatory compliance analyst with deep experience in:
1. Interpreting complex regulatory requirements
2. Identifying key compliance obligations
3. Extracting specific deadlines and metrics
4. Prioritizing requirements based on importance

Analyze the provided regulatory text and extract key information in a structured format.""",
            user_prompt_template="""Analyze the following regulatory text section and provide:

1. A concise summary focusing on key compliance requirements
2. A prioritized list of specific, actionable requirements
3. All deadlines, metrics, and numerical thresholds
4. Risk levels and compliance priorities

Regulatory text:
{text}

Provide the analysis in the following JSON format:
{{
    "summary": "Concise summary of key points",
    "key_requirements": [
        {{
            "requirement": "Specific requirement text",
            "priority": "high|medium|low",
            "category": "technical|operational|reporting|security",
            "risk_level": "critical|high|medium|low"
        }}
    ],
    "deadlines_and_metrics": [
        {{
            "type": "deadline|metric|threshold",
            "value": "extracted value",
            "context": "surrounding context",
            "timeframe": "immediate|short_term|long_term"
        }}
    ],
    "compliance_metadata": {{
        "primary_focus": "main focus area",
        "stakeholders": ["relevant stakeholders"],
        "implementation_complexity": "high|medium|low"
    }}
}}"""
        )
    }

    def __init__(self, config: OpenAIConfig):
        """
        Initialize LLM client.

        Args:
            config (OpenAIConfig): OpenAI configuration
            
        Raises:
            ValueError: If config is None or missing required fields
        """
        if not config:
            raise ValueError("OpenAIConfig is required")
        if not isinstance(config, OpenAIConfig):
            raise ValueError("config must be an instance of OpenAIConfig")
        if not config.api_key:
            raise ValueError("OpenAI API key is required")
            
        self.config = config
        self.client = AsyncOpenAI(api_key=config.api_key)
        self.prompts = self.DEFAULT_PROMPTS

    async def analyze_text(self, text: str, analysis_type: str = 'regulatory') -> Dict:
        """
        Analyze text using OpenAI's GPT model.

        Args:
            text (str): Text to analyze
            analysis_type (str): Type of analysis to perform
            
        Returns:
            Dict: Analysis results
            
        Raises:
            ValueError: If text is invalid or analysis type is unsupported
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        if analysis_type not in self.prompts:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")

        prompt = self._create_analysis_prompt(text, analysis_type)
        
        try:
            # Get completion with streaming
            response_chunks = []
            async for chunk in await self._get_streaming_completion(prompt):
                if chunk.choices[0].delta.content:
                    response_chunks.append(chunk.choices[0].delta.content)
            
            response_text = ''.join(response_chunks)
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")
                return self._format_unstructured_response(response_text)
                
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            raise

    async def _get_streaming_completion(self, prompt: AnalysisPrompt) -> ChatCompletion:
        """Get streaming completion from OpenAI."""
        return await self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt_template}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True
        )

    def _create_analysis_prompt(self, text: str, analysis_type: str) -> AnalysisPrompt:
        """
        Create the analysis prompt.

        Args:
            text (str): Text to analyze
            analysis_type (str): Type of analysis to perform

        Returns:
            AnalysisPrompt: Formatted prompt
        """
        if analysis_type not in self.prompts:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
        prompt = self.prompts[analysis_type]
        return AnalysisPrompt(
            system_prompt=prompt.system_prompt,
            user_prompt_template=prompt.user_prompt_template.format(text=text)
        )

    def _format_unstructured_response(self, text: str) -> Dict:
        """Format unstructured response into expected JSON structure."""
        return {
            "summary": text[:500],  # First 500 chars as summary
            "key_requirements": [{"requirement": text, "priority": "medium"}],
            "deadlines_and_metrics": [],
            "compliance_metadata": {
                "primary_focus": "unknown",
                "stakeholders": ["unknown"],
                "implementation_complexity": "medium"
            }
        }

    def add_custom_prompt(self, name: str, system_prompt: str, user_prompt_template: str) -> None:
        """
        Add a custom prompt template.

        Args:
            name (str): Name of the prompt template
            system_prompt (str): System prompt
            user_prompt_template (str): User prompt template
        """
        self.prompts[name] = AnalysisPrompt(
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template
        ) 

    async def analyze_requirements(self, text: str) -> List[str]:
        if not text:
            raise ValueError("Text cannot be empty")
            
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "Extract requirements from the text."},
                {"role": "user", "content": text}
            ]
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        return data.get('requirements', [])

    async def analyze_metrics(self, text: str) -> List[str]:
        if not text:
            raise ValueError("Text cannot be empty")
            
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "Extract metrics from the text."},
                {"role": "user", "content": text}
            ]
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        return data.get('metrics', []) 