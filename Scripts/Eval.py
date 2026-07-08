import pandas as pd
import json
import groq
from groq import Groq
import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_math_response(golden_answer, initial_response, refined_response, query):
    """
    Evaluate math responses using LLM with comprehensive metrics
    """
    
    evaluation_prompt = f"""
    You are an expert mathematics evaluator. Compare the following math solutions and provide a comprehensive evaluation.

    QUERY: {query}

    GOLDEN ANSWER (Ground Truth): {golden_answer}
    INITIAL RESPONSE: {initial_response}
    REFINED RESPONSE: {refined_response}

    Evaluate and return a JSON object with the following structure:

    {{
        "initial_response_evaluation": {{
            "correctness_score": 0-10,
            "simplicity_score": 0-10,
            "clarity_score": 0-10,
            "completeness_score": 0-10,
            "step_by_step_quality": 0-10,
            "overall_quality": 0-10,
            "strengths": ["list", "of", "strengths"],
            "weaknesses": ["list", "of", "weaknesses"]
        }},
        "refined_response_evaluation": {{
            "correctness_score": 0-10,
            "simplicity_score": 0-10,
            "clarity_score": 0-10,
            "completeness_score": 0-10,
            "step_by_step_quality": 0-10,
            "overall_quality": 0-10,
            "strengths": ["list", "of", "strengths"],
            "weaknesses": ["list", "of", "weaknesses"]
        }},
        "comparison_metrics": {{
            "improvement_score": 0-10,
            "refinement_effectiveness": 0-10,
            "human_feedback_importance": 0-10,
            "key_improvements": ["list", "of", "specific", "improvements"],
            "remaining_issues": ["list", "of", "remaining", "issues"]
        }},
        "quality_metrics": {{
            "mathematical_accuracy": 0-10,
            "explanation_quality": 0-10,
            "pedagogical_value": 0-10,
            "error_reduction": 0-10,
            "conceptual_clarity": 0-10
        }}
    }}

    SCORING GUIDELINES:
    - Correctness: How mathematically accurate is the solution?
    - Simplicity: How simple and straightforward is the explanation?
    - Clarity: How clear and easy to understand is the explanation?
    - Completeness: Does it cover all aspects of the problem?
    - Step-by-step Quality: How well are the steps explained?
    - Improvement Score: How much better is the refined response?
    - Refinement Effectiveness: How effective was the refinement process?
    - Human Feedback Importance: How crucial was human intervention?
    - Mathematical Accuracy: Alignment with mathematical principles
    - Explanation Quality: Quality of explanatory content
    - Pedagogical Value: Educational value for learning
    - Error Reduction: Reduction of errors/misconceptions
    - Conceptual Clarity: Clear explanation of underlying concepts

    Be strict but fair in your evaluation. Focus on mathematical accuracy and educational value.
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert mathematics educator and evaluator. You analyze math solutions for accuracy, clarity, and educational value. Always respond with valid JSON only."
                },
                {
                    "role": "user", 
                    "content": evaluation_prompt
                }
            ],
            temperature=0.1,
            max_tokens=4000,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"}
        )
        
        response_content = completion.choices[0].message.content
        evaluation_result = json.loads(response_content)
        return evaluation_result
        
    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        return None

def calculate_additional_metrics(evaluation_result):
    """
    Calculate derived metrics from the evaluation
    """
    if not evaluation_result:
        return None
    
    initial = evaluation_result['initial_response_evaluation']
    refined = evaluation_result['refined_response_evaluation']
    comparison = evaluation_result['comparison_metrics']
    
    additional_metrics = {
        # Improvement metrics
        "correctness_improvement": refined['correctness_score'] - initial['correctness_score'],
        "simplicity_improvement": refined['simplicity_score'] - initial['simplicity_score'],
        "overall_improvement": refined['overall_quality'] - initial['overall_quality'],
        
        # Percentage improvements
        "correctness_improvement_percent": ((refined['correctness_score'] - initial['correctness_score']) / max(initial['correctness_score'], 1)) * 100,
        "overall_improvement_percent": ((refined['overall_quality'] - initial['overall_quality']) / max(initial['overall_quality'], 1)) * 100,
        
        # Effectiveness scores
        "refinement_success_score": comparison['improvement_score'] * 0.3 + comparison['refinement_effectiveness'] * 0.4 + comparison['human_feedback_importance'] * 0.3,
        
        # Quality indices
        "initial_quality_index": sum([initial['correctness_score'], initial['clarity_score'], initial['completeness_score']]) / 3,
        "refined_quality_index": sum([refined['correctness_score'], refined['clarity_score'], refined['completeness_score']]) / 3,
        
        # Human feedback impact
        "human_feedback_impact": comparison['human_feedback_importance'] * (refined['overall_quality'] - initial['overall_quality']) / 10
    }
    
    return additional_metrics

def extract_summary_metrics(evaluation_result, additional_metrics):
    """
    Extract key summary metrics for quick analysis
    """
    if not evaluation_result:
        return None
    
    initial = evaluation_result['initial_response_evaluation']
    refined = evaluation_result['refined_response_evaluation']
    comparison = evaluation_result['comparison_metrics']
    
    return {
        'initial_quality': initial['overall_quality'],
        'refined_quality': refined['overall_quality'],
        'improvement_score': comparison['improvement_score'],
        'human_feedback_importance': comparison['human_feedback_importance'],
        'overall_improvement': additional_metrics.get('overall_improvement', 0) if additional_metrics else 0,
        'refinement_success_score': additional_metrics.get('refinement_success_score', 0) if additional_metrics else 0,
        'human_feedback_impact': additional_metrics.get('human_feedback_impact', 0) if additional_metrics else 0
    }

def print_summary(result_entry, entry_number):
    """
    Print a concise summary for each entry
    """
    summary = result_entry['summary_metrics']
    print(f"[OK] Entry {entry_number} completed:")
    print(f"   Initial: {summary['initial_quality']}/10 | "
          f"Refined: {summary['refined_quality']}/10 | "
          f"Improvement: +{summary['overall_improvement']:.1f} | "
          f"Human Impact: {summary['human_feedback_impact']:.1f}")

def process_batch(batch, starting_index):
    """
    Process a single batch of entries
    """
    batch_results = []
    
    for i, (index, row) in enumerate(batch.iterrows()):
        entry_number = starting_index + i
        print(f"\n--- Evaluating Entry {entry_number} ---")
        
        try:
            # Extract data with proper column names
            query = row.get('question', '')
            golden_answer = row.get('golden_answer', '')
            initial_response = row.get('initial_response', '')  # Note: lowercase 'i' based on your data
            refined_response = row.get('refined_response', '')
            
            # Skip if critical data is missing
            if not initial_response and not refined_response:
                logger.warning(f"Entry {entry_number}: Both responses empty, skipping")
                continue
            
            # Get LLM evaluation
            evaluation_result = evaluate_math_response(
                golden_answer, 
                initial_response, 
                refined_response, 
                query
            )
            
            if evaluation_result:
                # Calculate additional metrics
                additional_metrics = calculate_additional_metrics(evaluation_result)
                
                # Create result entry
                result_entry = {
                    'entry_id': entry_number,
                    'uuid': row.get('uuid', ''),
                    'subject': row.get('subject', ''),
                    'query_preview': str(query)[:100] + "..." if len(str(query)) > 100 else query,
                    'llm_evaluation': evaluation_result,
                    'calculated_metrics': additional_metrics,
                    'summary_metrics': extract_summary_metrics(evaluation_result, additional_metrics)
                }
                
                batch_results.append(result_entry)
                print_summary(result_entry, entry_number)
                
            else:
                logger.error(f"Entry {entry_number}: Evaluation failed")
                
        except Exception as e:
            logger.error(f"Entry {entry_number}: Error - {str(e)}")
            # Add error entry to maintain order
            error_entry = {
                'entry_id': entry_number,
                'error': str(e),
                'uuid': row.get('uuid', '') if 'row' in locals() else ''
            }
            batch_results.append(error_entry)
    
    return batch_results

def save_progress(results, batch_start):
    """
    Save progress to file after each batch
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_progress_batch_{batch_start}_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            # Convert to serializable format if needed
            serializable_results = []
            for result in results:
                if 'error' in result:
                    serializable_results.append(result)
                else:
                    serializable_results.append({
                        'entry_id': result['entry_id'],
                        'uuid': result['uuid'],
                        'subject': result['subject'],
                        'query_preview': result['query_preview'],
                        'summary_metrics': result['summary_metrics']
                    })
            
            json.dump(serializable_results, f, indent=2)
        print(f"[SAVED] Progress saved to {filename}")
        
    except Exception as e:
        logger.error(f"Failed to save progress: {e}")

def evaluate_batch_entries(df, batch_size=5, delay_between_batches=2):
    """
    Evaluate all entries in the dataframe in batches
    """
    results = []
    total_rows = len(df)
    
    print(f"Starting batch evaluation of {total_rows} entries...")
    print(f"Batch size: {batch_size}, Delay between batches: {delay_between_batches}s")
    print("=" * 80)
    
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch = df.iloc[batch_start:batch_end]
        
        print(f"\n[BATCH] Processing batch {batch_start//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}")
        print(f"Entries {batch_start} to {batch_end-1}")
        
        batch_results = process_batch(batch, batch_start)
        results.extend(batch_results)
        
        # Save progress after each batch
        save_progress(results, batch_start)
        
        # Delay between batches to avoid rate limits
        if batch_end < total_rows and delay_between_batches > 0:
            print(f"[WAIT] Waiting {delay_between_batches} seconds before next batch...")
            time.sleep(delay_between_batches)
    
    return results

def generate_final_report(results):
    """
    Generate a comprehensive final report after all batches
    """
    successful_results = [r for r in results if 'error' not in r]
    
    if not successful_results:
        print("No successful evaluations to report")
        return
    
    print("\n" + "="*80)
    print("FINAL EVALUATION REPORT")
    print("="*80)
    
    # Calculate averages
    initial_qualities = [r['summary_metrics']['initial_quality'] for r in successful_results]
    refined_qualities = [r['summary_metrics']['refined_quality'] for r in successful_results]
    improvements = [r['summary_metrics']['overall_improvement'] for r in successful_results]
    human_impacts = [r['summary_metrics']['human_feedback_impact'] for r in successful_results]
    
    print(f"Total Entries Evaluated: {len(successful_results)}")
    print(f"Average Initial Quality: {sum(initial_qualities)/len(initial_qualities):.2f}/10")
    print(f"Average Refined Quality: {sum(refined_qualities)/len(refined_qualities):.2f}/10")
    print(f"Average Improvement: {sum(improvements)/len(improvements):.2f} points")
    print(f"Average Human Impact: {sum(human_impacts)/len(human_impacts):.2f}")
    print(f"Success Rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
    
    # Quality distribution
    quality_ranges = {'0-3': 0, '4-6': 0, '7-8': 0, '9-10': 0}
    for quality in refined_qualities:
        if quality <= 3: quality_ranges['0-3'] += 1
        elif quality <= 6: quality_ranges['4-6'] += 1
        elif quality <= 8: quality_ranges['7-8'] += 1
        else: quality_ranges['9-10'] += 1
    
    print(f"\nRefined Quality Distribution:")
    for range_name, count in quality_ranges.items():
        percentage = count/len(refined_qualities)*100
        print(f"  {range_name}/10: {count} entries ({percentage:.1f}%)")
    
    # Save detailed final results
    final_filename = f"final_evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(final_filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n[SAVED] Detailed results saved to {final_filename}")

def evaluate_all_entries(df, batch_size=5, delay_between_batches=2):
    """
    Main function to evaluate all entries in batches
    """
    print("STARTING comprehensive evaluation of all entries...")
    print(f"Dataset size: {len(df)} entries")
    print(f"Batch configuration: {batch_size} entries per batch, {delay_between_batches}s delay")
    
    # Verify column names - using lowercase 'initial_response' based on your data
    required_columns = ['question', 'golden_answer', 'initial_response', 'refined_response']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"[ERROR] Missing columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    print("[OK] All required columns present")
    
    # Start batch evaluation
    start_time = time.time()
    results = evaluate_batch_entries(df, batch_size, delay_between_batches)
    end_time = time.time()
    
    # Generate final report
    generate_final_report(results)
    
    print(f"\n[TIME] Total evaluation time: {(end_time - start_time)/60:.2f} minutes")
    print("[DONE] Evaluation complete!")
    
    return results

if __name__ == "__main__":
    # Load your evaluation dataset
    eval_df = pd.read_csv("D:\\programming\\python\\Math_prof\\tests\\EvalSet.csv", encoding='latin-1')
    print(f"Dataset loaded with {len(eval_df)} entries")
    print("\nFirst entry preview:")
    print(eval_df.iloc[0])
    
    # Run evaluation on all entries
    results = evaluate_all_entries(eval_df, batch_size=4, delay_between_batches=120)
    
    # Save final results
    if results:
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Final results saved to 'evaluation_results.json'")