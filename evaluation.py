"""
Evaluation Module for Multi-Language RAG System
Provides testing and benchmarking capabilities
"""
import logging
import time
import json
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path

import config
from rag_engine import RAGEngine

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """
    Evaluates the performance and quality of the RAG system.
    Provides metrics for retrieval accuracy, response quality, and system performance.
    """
    
    def __init__(self, rag_engine: RAGEngine):
        """
        Initialize the evaluator.
        
        Args:
            rag_engine (RAGEngine): Initialized RAG engine instance
        """
        self.rag_engine = rag_engine
        self.test_results = []
        
    def run_basic_evaluation(self) -> Dict[str, Any]:
        """
        Run basic system evaluation with predefined test cases.
        
        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info("Starting basic system evaluation...")
        
        # Predefined test cases in multiple languages
        test_cases = [
            {
                'id': 'health_001',
                'query': 'What are the symptoms of diabetes?',
                'language': 'en',
                'expected_keywords': ['symptoms', 'diabetes', 'blood', 'glucose'],
                'category': 'symptoms'
            },
            {
                'id': 'health_002',
                'query': '¿Cuáles son los síntomas de la diabetes?',
                'language': 'es',
                'expected_keywords': ['síntomas', 'diabetes', 'glucosa'],
                'category': 'symptoms'
            },
            {
                'id': 'health_003',
                'query': 'Quels sont les symptômes du diabète?',
                'language': 'fr',
                'expected_keywords': ['symptômes', 'diabète', 'glycémie'],
                'category': 'symptoms'
            },
            {
                'id': 'health_004',
                'query': 'Was sind die Symptome von Diabetes?',
                'language': 'de',
                'expected_keywords': ['Symptome', 'Diabetes', 'Blutzucker'],
                'category': 'symptoms'
            },
            {
                'id': 'health_005',
                'query': '糖尿病の症状は何ですか？',
                'language': 'ja',
                'expected_keywords': ['症状', '糖尿病'],
                'category': 'symptoms'
            },
            {
                'id': 'health_006',
                'query': 'What treatments are available for diabetes?',
                'language': 'en',
                'expected_keywords': ['treatment', 'medication', 'insulin', 'diet'],
                'category': 'treatment'
            },
            {
                'id': 'health_007',
                'query': '¿Qué tratamientos están disponibles para la diabetes?',
                'language': 'es',
                'expected_keywords': ['tratamiento', 'medicamento', 'insulina'],
                'category': 'treatment'
            }
        ]
        
        results = {
            'total_tests': len(test_cases),
            'successful_tests': 0,
            'failed_tests': 0,
            'average_response_time': 0.0,
            'language_coverage': set(),
            'category_performance': {},
            'detailed_results': []
        }
        
        total_response_time = 0.0
        
        for test_case in test_cases:
            logger.info(f"Testing case {test_case['id']}: {test_case['query']}")
            
            start_time = time.time()
            
            try:
                # Run query
                response = self.rag_engine.query(
                    test_case['query'],
                    target_language=test_case['language']
                )
                
                response_time = time.time() - start_time
                total_response_time += response_time
                
                # Evaluate response
                evaluation = self._evaluate_response(test_case, response)
                
                # Record results
                test_result = {
                    'test_id': test_case['id'],
                    'query': test_case['query'],
                    'language': test_case['language'],
                    'category': test_case['category'],
                    'response': response,
                    'evaluation': evaluation,
                    'response_time': response_time,
                    'success': evaluation['overall_score'] >= 0.6
                }
                
                results['detailed_results'].append(test_result)
                
                # Update statistics
                if test_result['success']:
                    results['successful_tests'] += 1
                else:
                    results['failed_tests'] += 1
                
                results['language_coverage'].add(test_case['language'])
                
                # Update category performance
                category = test_case['category']
                if category not in results['category_performance']:
                    results['category_performance'][category] = {
                        'total': 0,
                        'successful': 0,
                        'average_score': 0.0
                    }
                
                results['category_performance'][category]['total'] += 1
                if test_result['success']:
                    results['category_performance'][category]['successful'] += 1
                
                # Calculate running average for category
                current_avg = results['category_performance'][category]['average_score']
                current_total = results['category_performance'][category]['total']
                new_score = evaluation['overall_score']
                results['category_performance'][category]['average_score'] = (
                    (current_avg * (current_total - 1) + new_score) / current_total
                )
                
                logger.info(f"Test {test_case['id']} completed with score: {evaluation['overall_score']:.3f}")
                
            except Exception as e:
                logger.error(f"Error in test case {test_case['id']}: {e}")
                results['failed_tests'] += 1
                
                # Add failed result
                test_result = {
                    'test_id': test_case['id'],
                    'query': test_case['query'],
                    'language': test_case['language'],
                    'category': test_case['category'],
                    'response': None,
                    'evaluation': {'overall_score': 0.0, 'error': str(e)},
                    'response_time': 0.0,
                    'success': False
                }
                results['detailed_results'].append(test_result)
        
        # Calculate final statistics
        if results['total_tests'] > 0:
            results['average_response_time'] = total_response_time / results['total_tests']
            results['success_rate'] = results['successful_tests'] / results['total_tests']
        
        results['language_coverage'] = list(results['language_coverage'])
        
        logger.info(f"Basic evaluation completed. Success rate: {results.get('success_rate', 0):.2%}")
        
        return results
    
    def _evaluate_response(self, test_case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of a response.
        
        Args:
            test_case (Dict[str, Any]): Test case information
            response (Dict[str, Any]): System response
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        try:
            if response['status'] != 'success':
                return {
                    'overall_score': 0.0,
                    'relevance_score': 0.0,
                    'completeness_score': 0.0,
                    'language_accuracy_score': 0.0,
                    'error': 'Response status not successful'
                }
            
            # Relevance score (based on similarity scores)
            relevance_score = 0.0
            if response.get('sources'):
                avg_similarity = sum(s['similarity_score'] for s in response['sources']) / len(response['sources'])
                relevance_score = min(1.0, avg_similarity * 1.2)  # Boost similarity score
            
            # Completeness score (based on answer length and content)
            answer = response.get('answer', '')
            completeness_score = min(1.0, len(answer) / 100)  # Normalize by expected length
            
            # Language accuracy score
            language_accuracy_score = 1.0
            if response.get('query_language') != response.get('response_language'):
                # Check if translation was needed and successful
                if 'translation_metadata' in response:
                    language_accuracy_score = 0.8  # Slight penalty for translation
            
            # Keyword presence score
            keyword_score = 0.0
            expected_keywords = test_case.get('expected_keywords', [])
            if expected_keywords and answer:
                answer_lower = answer.lower()
                found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
                keyword_score = found_keywords / len(expected_keywords)
            
            # Overall score (weighted average)
            overall_score = (
                relevance_score * 0.3 +
                completeness_score * 0.2 +
                language_accuracy_score * 0.2 +
                keyword_score * 0.3
            )
            
            return {
                'overall_score': overall_score,
                'relevance_score': relevance_score,
                'completeness_score': completeness_score,
                'language_accuracy_score': language_accuracy_score,
                'keyword_score': keyword_score,
                'detailed_breakdown': {
                    'relevance_weight': 0.3,
                    'completeness_weight': 0.2,
                    'language_weight': 0.2,
                    'keyword_weight': 0.3
                }
            }
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return {
                'overall_score': 0.0,
                'relevance_score': 0.0,
                'completeness_score': 0.0,
                'language_accuracy_score': 0.0,
                'error': str(e)
            }
    
    def run_performance_benchmark(self, num_queries: int = 50) -> Dict[str, Any]:
        """
        Run performance benchmarking with multiple queries.
        
        Args:
            num_queries (int): Number of queries to test
            
        Returns:
            Dict[str, Any]: Performance benchmark results
        """
        logger.info(f"Starting performance benchmark with {num_queries} queries...")
        
        # Generate test queries
        test_queries = self._generate_test_queries(num_queries)
        
        results = {
            'total_queries': num_queries,
            'response_times': [],
            'memory_usage': [],
            'success_rates': [],
            'language_distribution': {},
            'performance_metrics': {}
        }
        
        start_time = time.time()
        
        for i, query in enumerate(test_queries):
            logger.info(f"Benchmark query {i+1}/{num_queries}")
            
            query_start = time.time()
            
            try:
                response = self.rag_engine.query(query['text'], query['language'])
                query_time = time.time() - query_start
                
                results['response_times'].append(query_time)
                
                # Record language distribution
                lang = query['language']
                results['language_distribution'][lang] = results['language_distribution'].get(lang, 0) + 1
                
                # Record success
                success = response['status'] == 'success'
                results['success_rates'].append(success)
                
            except Exception as e:
                logger.error(f"Error in benchmark query {i+1}: {e}")
                results['response_times'].append(0.0)
                results['success_rates'].append(False)
        
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        if results['response_times']:
            results['performance_metrics'] = {
                'total_execution_time': total_time,
                'average_response_time': sum(results['response_times']) / len(results['response_times']),
                'min_response_time': min(results['response_times']),
                'max_response_time': max(results['response_times']),
                'queries_per_second': num_queries / total_time,
                'success_rate': sum(results['success_rates']) / len(results['success_rates']),
                'throughput': num_queries / total_time
            }
        
        logger.info(f"Performance benchmark completed in {total_time:.2f} seconds")
        
        return results
    
    def _generate_test_queries(self, num_queries: int) -> List[Dict[str, Any]]:
        """
        Generate test queries for benchmarking.
        
        Args:
            num_queries (int): Number of queries to generate
            
        Returns:
            List[Dict[str, Any]]: List of test queries
        """
        # Healthcare-related query templates
        query_templates = [
            "What are the symptoms of {condition}?",
            "How is {condition} treated?",
            "What causes {condition}?",
            "What are the risk factors for {condition}?",
            "How is {condition} diagnosed?",
            "What medications are used for {condition}?",
            "What lifestyle changes help with {condition}?",
            "What are the complications of {condition}?",
            "How can {condition} be prevented?",
            "What tests are done for {condition}?"
        ]
        
        # Common health conditions
        conditions = [
            "diabetes", "hypertension", "asthma", "arthritis", "depression",
            "anxiety", "heart disease", "cancer", "obesity", "osteoporosis"
        ]
        
        # Languages to test
        languages = list(config.SUPPORTED_LANGUAGES.keys())
        
        queries = []
        for i in range(num_queries):
            template = query_templates[i % len(query_templates)]
            condition = conditions[i % len(conditions)]
            language = languages[i % len(languages)]
            
            # Generate query in the target language (simplified)
            if language == 'en':
                query_text = template.format(condition=condition)
            else:
                # For other languages, use a simple translation approach
                # In a real system, you might use proper translation
                query_text = f"{template.format(condition=condition)} ({language})"
            
            queries.append({
                'text': query_text,
                'language': language,
                'template': template,
                'condition': condition
            })
        
        return queries
    
    def generate_report(self, evaluation_results: Dict[str, Any], 
                       output_file: str = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results (Dict[str, Any]): Evaluation results
            output_file (str): Optional output file path
            
        Returns:
            str: Generated report content
        """
        report = []
        report.append("=" * 80)
        report.append("MULTI-LANGUAGE RAG SYSTEM EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Tests: {evaluation_results.get('total_tests', 0)}")
        report.append(f"Successful Tests: {evaluation_results.get('successful_tests', 0)}")
        report.append(f"Failed Tests: {evaluation_results.get('failed_tests', 0)}")
        report.append(f"Success Rate: {evaluation_results.get('success_rate', 0):.2%}")
        report.append(f"Average Response Time: {evaluation_results.get('average_response_time', 0):.3f}s")
        report.append("")
        
        # Language coverage
        report.append("LANGUAGE COVERAGE")
        report.append("-" * 40)
        languages = evaluation_results.get('language_coverage', [])
        for lang in languages:
            lang_name = config.SUPPORTED_LANGUAGES.get(lang, lang)
            report.append(f"- {lang_name} ({lang})")
        report.append("")
        
        # Category performance
        report.append("CATEGORY PERFORMANCE")
        report.append("-" * 40)
        category_perf = evaluation_results.get('category_performance', {})
        for category, perf in category_perf.items():
            success_rate = perf['successful'] / perf['total'] if perf['total'] > 0 else 0
            report.append(f"{category.capitalize()}: {success_rate:.2%} ({perf['successful']}/{perf['total']})")
        report.append("")
        
        # Performance metrics (if available)
        if 'performance_metrics' in evaluation_results:
            report.append("PERFORMANCE METRICS")
            report.append("-" * 40)
            perf = evaluation_results['performance_metrics']
            report.append(f"Queries per Second: {perf.get('queries_per_second', 0):.2f}")
            report.append(f"Throughput: {perf.get('throughput', 0):.2f} queries/min")
            report.append(f"Min Response Time: {perf.get('min_response_time', 0):.3f}s")
            report.append(f"Max Response Time: {perf.get('max_response_time', 0):.3f}s")
            report.append("")
        
        # Detailed results
        report.append("DETAILED TEST RESULTS")
        report.append("-" * 40)
        detailed_results = evaluation_results.get('detailed_results', [])
        for result in detailed_results[:10]:  # Show first 10 results
            report.append(f"Test ID: {result['test_id']}")
            report.append(f"Query: {result['query']}")
            report.append(f"Language: {result['language']}")
            report.append(f"Success: {result['success']}")
            report.append(f"Score: {result['evaluation'].get('overall_score', 0):.3f}")
            report.append(f"Response Time: {result['response_time']:.3f}s")
            report.append("")
        
        if len(detailed_results) > 10:
            report.append(f"... and {len(detailed_results) - 10} more results")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        success_rate = evaluation_results.get('success_rate', 0)
        
        if success_rate >= 0.8:
            report.append("✅ System performance is excellent!")
            report.append("Recommendations:")
            report.append("- Consider adding more complex test cases")
            report.append("- Monitor performance under higher load")
        elif success_rate >= 0.6:
            report.append("⚠️ System performance is good but has room for improvement")
            report.append("Recommendations:")
            report.append("- Review failed test cases for patterns")
            report.append("- Optimize embedding and retrieval algorithms")
            report.append("- Consider fine-tuning similarity thresholds")
        else:
            report.append("❌ System performance needs significant improvement")
            report.append("Recommendations:")
            report.append("- Review system configuration")
            report.append("- Check document quality and indexing")
            report.append("- Verify embedding model performance")
            report.append("- Consider system architecture improvements")
        
        report.append("")
        report.append("=" * 80)
        
        report_content = "\n".join(report)
        
        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"Evaluation report saved to: {output_file}")
            except Exception as e:
                logger.error(f"Error saving report to {output_file}: {e}")
        
        return report_content
    
    def save_results(self, results: Dict[str, Any], output_file: str) -> bool:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results (Dict[str, Any]): Evaluation results
            output_file (str): Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert sets to lists for JSON serialization
            json_results = results.copy()
            if 'language_coverage' in json_results and isinstance(json_results['language_coverage'], set):
                json_results['language_coverage'] = list(json_results['language_coverage'])
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Evaluation results saved to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}")
            return False

def main():
    """Main function for running evaluations."""
    print("Multi-Language RAG System Evaluator")
    print("=" * 50)
    
    try:
        # Initialize RAG engine
        print("Initializing RAG Engine...")
        rag_engine = RAGEngine()
        
        # Add sample documents if needed
        print("Adding sample documents...")
        result = rag_engine.add_sample_documents()
        if result['status'] != 'success':
            print(f"Warning: Could not add sample documents: {result.get('message', 'Unknown error')}")
        
        # Initialize evaluator
        evaluator = RAGEvaluator(rag_engine)
        
        # Run basic evaluation
        print("\nRunning basic evaluation...")
        basic_results = evaluator.run_basic_evaluation()
        
        # Run performance benchmark
        print("\nRunning performance benchmark...")
        perf_results = evaluator.run_performance_benchmark(num_queries=20)
        
        # Combine results
        combined_results = {
            'basic_evaluation': basic_results,
            'performance_benchmark': perf_results,
            'evaluation_timestamp': time.time()
        }
        
        # Generate report
        print("\nGenerating evaluation report...")
        report = evaluator.generate_report(basic_results)
        print("\n" + report)
        
        # Save results
        results_file = "evaluation_results.json"
        if evaluator.save_results(combined_results, results_file):
            print(f"\nResults saved to: {results_file}")
        
        # Save report
        report_file = "evaluation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {report_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        logger.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()
