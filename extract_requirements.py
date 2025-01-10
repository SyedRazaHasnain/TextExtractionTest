#!/usr/bin/env python3
"""
Regulatory Text Analyzer

This script processes regulatory text files and extracts key business requirements
using either NLP techniques or OpenAI's API for analysis.
"""

import argparse
import json
import logging
from typing import Dict, List

from src.config import ConfigManager, AnalysisMode
from src.text_processor import TextProcessor
from src.text_analyzer import TextAnalyzer
from src.llm_client import LLMClient

class RegulatoryTextAnalyzer:
    """Orchestrates the analysis of regulatory text documents."""

    def __init__(self, input_file: str, output_file: str, analysis_mode: str = None):
        """
        Initialize the analyzer.

        Args:
            input_file (str): Path to input file
            output_file (str): Path to output file
            analysis_mode (str, optional): Analysis mode ('simulated' or 'openai')
            
        Raises:
            ValueError: If input parameters are invalid
            FileNotFoundError: If input file doesn't exist
        """
        if not input_file or not isinstance(input_file, str):
            raise ValueError("Input file path is required and must be a string")
        if not output_file or not isinstance(output_file, str):
            raise ValueError("Output file path is required and must be a string")
        if analysis_mode and analysis_mode not in ['simulated', 'openai']:
            raise ValueError("Analysis mode must be either 'simulated' or 'openai'")
            
        # Initialize configuration
        config_manager = ConfigManager()
        self.app_config = config_manager.get_app_config(input_file, output_file, analysis_mode)

        # Configure logging
        logging.basicConfig(
            level=self.app_config.log_level,
            format=self.app_config.log_format
        )
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.text_processor = TextProcessor()
        
        # Initialize appropriate analyzer based on mode
        if self.app_config.analysis_mode == AnalysisMode.OPENAI:
            openai_config = config_manager.get_openai_config()
            if not openai_config:
                raise ValueError("OpenAI configuration is required for OpenAI analysis mode")
            self.analyzer = LLMClient(openai_config)
            self.logger.info("Using OpenAI for text analysis")
        else:
            self.analyzer = TextAnalyzer()
            self.logger.info("Using simulated NLP for text analysis")

    def _read_document(self) -> str:
        """Read the regulatory document from file."""
        try:
            with open(self.app_config.input_file, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            self.logger.error(f"Failed to read document: {str(e)}")
            raise

    def _save_analysis(self, analysis_results: List[Dict]) -> None:
        """Save analysis results to file."""
        try:
            with open(self.app_config.output_file, 'w', encoding='utf-8') as file:
                json.dump(analysis_results, file, indent=2)
            self.logger.info(f"Analysis results saved to {self.app_config.output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save analysis results: {str(e)}")
            raise

    async def _analyze_section(self, section_num: int, section_text: str) -> Dict:
        """
        Analyze a single section of regulatory text.

        Args:
            section_num (int): Section number
            section_text (str): Section content

        Returns:
            Dict: Analysis results for the section
        """
        try:
            # Get text analysis and metrics
            if isinstance(self.analyzer, LLMClient):
                analysis = await self.analyzer.analyze_text(section_text)
            else:
                analysis = self.analyzer.analyze_text(section_text)
                
            metrics = self.text_processor.get_text_metrics(section_text)

            return {
                'section_number': section_num,
                'original_text': section_text,
                'analysis': analysis,
                'metadata': metrics
            }
        except Exception as e:
            self.logger.error(f"Failed to analyze section {section_num}: {str(e)}")
            raise

    async def analyze_document(self) -> None:
        """
        Process the complete document:
        1. Read the document
        2. Segment into sections
        3. Analyze each section
        4. Save results
        """
        try:
            # Read and segment document
            document = self._read_document()
            sections = self.text_processor.segment_text(document)

            # Process sections
            if isinstance(self.analyzer, LLMClient):
                # Process sections in parallel if using OpenAI
                import asyncio
                tasks = [
                    self._analyze_section(section_num, section_text)
                    for section_num, section_text in sections
                ]
                results = await asyncio.gather(*tasks)
            else:
                # Process sections sequentially if using simulated analysis
                results = []
                for section_num, section_text in sections:
                    result = await self._analyze_section(section_num, section_text)
                    results.append(result)

            # Save analysis results
            self._save_analysis(results)
            self.logger.info("Document analysis completed successfully")

        except Exception as e:
            self.logger.error(f"Document analysis failed: {str(e)}")
            raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze regulatory text documents.')
    parser.add_argument('--input', default='regulations.txt', help='Input file path')
    parser.add_argument('--output', default='extracted_requirements.json', help='Output file path')
    parser.add_argument('--mode', choices=['simulated', 'openai'], default='simulated',
                       help='Analysis mode (simulated or openai)')
    return parser.parse_args()

async def main():
    """Main execution function."""
    try:
        args = parse_args()
        analyzer = RegulatoryTextAnalyzer(
            input_file=args.input,
            output_file=args.output,
            analysis_mode=args.mode
        )
        await analyzer.analyze_document()
    except Exception as e:
        logging.error(f"Application failed: {str(e)}")
        raise

if __name__ == '__main__':
    import asyncio
    asyncio.run(main()) 