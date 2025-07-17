import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import hashlib
import chardet
import re
from pathlib import Path

class TMXAnalyzer:
    def __init__(self, file_paths):
        """
        Initialize analyzer with list of TMX file paths
        file_paths: list of strings, paths to TMX files
        """
        self.file_paths = file_paths
        self.analysis_results = {}
        
    def analyze_all_files(self):
        """
        Run complete analysis on all TMX files
        """
        print("=" * 60)
        print("TMX DATA QUALITY ANALYSIS")
        print("=" * 60)
        
        for i, file_path in enumerate(self.file_paths, 1):
            print(f"\n{'='*20} FILE {i}: {Path(file_path).name} {'='*20}")
            
            try:
                # Check encoding first
                encoding_info = self.check_encoding(file_path)
                print(f"File encoding: {encoding_info['encoding']} (confidence: {encoding_info['confidence']:.2f})")
                
                # Parse TMX file
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # Extract translation data
                translation_data = self.extract_translation_data(root)
                
                # Perform all analyses
                results = {
                    'file_path': file_path,
                    'encoding': encoding_info,
                    'basic_stats': self.get_basic_stats(translation_data),
                    'length_ratios': self.analyze_length_ratios(translation_data),
                    'empty_segments': self.count_empty_segments(translation_data),
                    'duplicates': self.detect_duplicates(translation_data),
                    'language_pairs': self.analyze_language_pairs(translation_data)
                }
                
                self.analysis_results[file_path] = results
                self.print_file_analysis(results)
                
            except Exception as e:
                print(f"ERROR analyzing {file_path}: {str(e)}")
                continue
        
        # Print summary comparison
        self.print_summary_comparison()
        
        return self.analysis_results
    
    def check_encoding(self, file_path):
        """
        Check file encoding
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return {
                    'encoding': result['encoding'],
                    'confidence': result['confidence']
                }
        except Exception as e:
            return {
                'encoding': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def extract_translation_data(self, root):
        """
        Extract translation data from TMX root
        """
        translation_data = []
        
        # Find all translation units
        translation_units = root.findall('.//tu')
        
        for tu in translation_units:
            tu_data = {
                'tu_id': tu.get('tuid', ''),
                'segments': {}
            }
            
            # Find all translation unit variants (tuvs)
            tuvs = tu.findall('tuv')
            
            for tuv in tuvs:
                # Get language
                lang = (tuv.get('xml:lang') or 
                       tuv.get('lang') or 
                       tuv.get('{http://www.w3.org/XML/1998/namespace}lang') or 
                       'unknown')
                
                # Get text content
                seg = tuv.find('seg')
                if seg is not None:
                    text = ''.join(seg.itertext()).strip()
                    if not text:
                        text = seg.text or ""
                else:
                    text = ""
                
                tu_data['segments'][lang] = text
            
            translation_data.append(tu_data)
        
        return translation_data
    
    def get_basic_stats(self, translation_data):
        """
        Get basic statistics about the translation data
        """
        total_tus = len(translation_data)
        languages = set()
        
        for tu in translation_data:
            languages.update(tu['segments'].keys())
        
        return {
            'total_translation_units': total_tus,
            'languages_found': sorted(list(languages)),
            'language_count': len(languages)
        }
    
    def analyze_length_ratios(self, translation_data):
        """
        Analyze length ratios between source and target segments
        """
        ratios = defaultdict(list)
        char_lengths = defaultdict(list)
        word_lengths = defaultdict(list)
        
        # Assume first language is source (or identify most common)
        if not translation_data:
            return {}
        
        # Find source language (usually most segments or explicitly defined)
        lang_counts = Counter()
        for tu in translation_data:
            for lang in tu['segments'].keys():
                if tu['segments'][lang].strip():  # Only count non-empty
                    lang_counts[lang] += 1
        
        if not lang_counts:
            return {'error': 'No valid segments found'}
        
        # Assume source is most frequent language or first alphabetically
        source_lang = lang_counts.most_common(1)[0][0]
        
        for tu in translation_data:
            segments = tu['segments']
            
            if source_lang not in segments or not segments[source_lang].strip():
                continue
            
            source_text = segments[source_lang].strip()
            source_chars = len(source_text)
            source_words = len(source_text.split())
            
            for lang, text in segments.items():
                if lang == source_lang or not text.strip():
                    continue
                
                target_chars = len(text.strip())
                target_words = len(text.strip().split())
                
                if source_chars > 0:
                    char_ratio = target_chars / source_chars
                    ratios[f"{source_lang}->{lang}"].append(char_ratio)
                    char_lengths[f"{source_lang}_chars"].append(source_chars)
                    char_lengths[f"{lang}_chars"].append(target_chars)
                
                if source_words > 0:
                    word_ratio = target_words / source_words
                    word_lengths[f"{source_lang}_words"].append(source_words)
                    word_lengths[f"{lang}_words"].append(target_words)
        
        # Calculate statistics
        ratio_stats = {}
        for lang_pair, ratio_list in ratios.items():
            if ratio_list:
                ratio_stats[lang_pair] = {
                    'mean_ratio': np.mean(ratio_list),
                    'median_ratio': np.median(ratio_list),
                    'std_ratio': np.std(ratio_list),
                    'min_ratio': np.min(ratio_list),
                    'max_ratio': np.max(ratio_list),
                    'count': len(ratio_list)
                }
        
        return {
            'source_language': source_lang,
            'character_ratios': ratio_stats,
            'length_distributions': {
                'characters': {k: {
                    'mean': np.mean(v),
                    'median': np.median(v),
                    'std': np.std(v)
                } for k, v in char_lengths.items()},
                'words': {k: {
                    'mean': np.mean(v),
                    'median': np.median(v),
                    'std': np.std(v)
                } for k, v in word_lengths.items()}
            }
        }
    
    def count_empty_segments(self, translation_data):
        """
        Count empty segments by language
        """
        empty_counts = defaultdict(int)
        total_counts = defaultdict(int)
        
        for tu in translation_data:
            for lang, text in tu['segments'].items():
                total_counts[lang] += 1
                if not text or not text.strip():
                    empty_counts[lang] += 1
        
        # Calculate percentages
        empty_percentages = {}
        for lang in total_counts:
            empty_percentages[lang] = {
                'empty_count': empty_counts[lang],
                'total_count': total_counts[lang],
                'empty_percentage': (empty_counts[lang] / total_counts[lang]) * 100 if total_counts[lang] > 0 else 0
            }
        
        return empty_percentages
    
    def detect_duplicates(self, translation_data):
        """
        Detect duplicate segments within each language
        """
        duplicate_stats = {}
        
        for tu in translation_data:
            for lang, text in tu['segments'].items():
                if lang not in duplicate_stats:
                    duplicate_stats[lang] = {
                        'text_counts': Counter(),
                        'total_segments': 0
                    }
                
                duplicate_stats[lang]['total_segments'] += 1
                
                # Normalize text for duplicate detection
                normalized_text = text.strip().lower()
                if normalized_text:
                    duplicate_stats[lang]['text_counts'][normalized_text] += 1
        
        # Calculate duplicate statistics
        duplicate_summary = {}
        for lang, stats in duplicate_stats.items():
            text_counts = stats['text_counts']
            total_segments = stats['total_segments']
            
            # Find duplicates (count > 1)
            duplicates = {text: count for text, count in text_counts.items() if count > 1}
            
            duplicate_summary[lang] = {
                'total_segments': total_segments,
                'unique_texts': len(text_counts),
                'duplicate_texts': len(duplicates),
                'duplicate_percentage': (len(duplicates) / len(text_counts)) * 100 if text_counts else 0,
                'most_common_duplicates': text_counts.most_common(5),
                'total_duplicate_instances': sum(count - 1 for count in duplicates.values())
            }
        
        return duplicate_summary
    
    def analyze_language_pairs(self, translation_data):
        """
        Analyze language pair completeness
        """
        language_pair_matrix = defaultdict(int)
        
        for tu in translation_data:
            languages = [lang for lang, text in tu['segments'].items() if text.strip()]
            
            # Count all possible pairs
            for i, lang1 in enumerate(languages):
                for lang2 in languages[i+1:]:
                    pair = f"{lang1}<->{lang2}"
                    language_pair_matrix[pair] += 1
        
        return dict(language_pair_matrix)
    
    def print_file_analysis(self, results):
        """
        Print analysis results for a single file
        """
        print(f"\n📊 BASIC STATISTICS:")
        basic = results['basic_stats']
        print(f"  • Total Translation Units: {basic['total_translation_units']:,}")
        print(f"  • Languages Found: {', '.join(basic['languages_found'])}")
        print(f"  • Language Count: {basic['language_count']}")
        
        print(f"\n📏 LENGTH RATIO ANALYSIS:")
        if 'error' in results['length_ratios']:
            print(f"  ❌ Error: {results['length_ratios']['error']}")
        else:
            length_data = results['length_ratios']
            print(f"  • Source Language: {length_data['source_language']}")
            
            for lang_pair, stats in length_data['character_ratios'].items():
                print(f"  • {lang_pair}:")
                print(f"    - Mean ratio: {stats['mean_ratio']:.2f}")
                print(f"    - Median ratio: {stats['median_ratio']:.2f}")
                print(f"    - Std deviation: {stats['std_ratio']:.2f}")
                print(f"    - Range: {stats['min_ratio']:.2f} - {stats['max_ratio']:.2f}")
                print(f"    - Sample size: {stats['count']:,}")
        
        print(f"\n❌ EMPTY SEGMENTS:")
        for lang, empty_data in results['empty_segments'].items():
            print(f"  • {lang}: {empty_data['empty_count']:,} empty / {empty_data['total_count']:,} total ({empty_data['empty_percentage']:.1f}%)")
        
        print(f"\n🔄 DUPLICATE DETECTION:")
        for lang, dup_data in results['duplicates'].items():
            print(f"  • {lang}:")
            print(f"    - Total segments: {dup_data['total_segments']:,}")
            print(f"    - Unique texts: {dup_data['unique_texts']:,}")
            print(f"    - Duplicate texts: {dup_data['duplicate_texts']:,} ({dup_data['duplicate_percentage']:.1f}%)")
            print(f"    - Duplicate instances: {dup_data['total_duplicate_instances']:,}")
            
            if dup_data['most_common_duplicates']:
                print(f"    - Most common duplicates:")
                for text, count in dup_data['most_common_duplicates'][:3]:
                    display_text = text[:50] + "..." if len(text) > 50 else text
                    print(f"      '{display_text}' ({count} times)")
        
        print(f"\n🔗 LANGUAGE PAIR COMPLETENESS:")
        for pair, count in results['language_pairs'].items():
            print(f"  • {pair}: {count:,} complete pairs")
    
    def print_summary_comparison(self):
        """
        Print comparison summary across all files
        """
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON ACROSS ALL FILES")
        print(f"{'='*60}")
        
        # Create comparison table
        comparison_data = []
        
        for file_path, results in self.analysis_results.items():
            file_name = Path(file_path).name
            basic = results['basic_stats']
            
            # Calculate quality scores
            empty_scores = results['empty_segments']
            avg_empty_rate = np.mean([data['empty_percentage'] for data in empty_scores.values()])
            
            duplicate_scores = results['duplicates']
            avg_duplicate_rate = np.mean([data['duplicate_percentage'] for data in duplicate_scores.values()])
            
            comparison_data.append({
                'File': file_name,
                'Translation Units': basic['total_translation_units'],
                'Languages': basic['language_count'],
                'Avg Empty Rate (%)': f"{avg_empty_rate:.1f}",
                'Avg Duplicate Rate (%)': f"{avg_duplicate_rate:.1f}",
                'Encoding': results['encoding']['encoding'],
                'Encoding Confidence': f"{results['encoding']['confidence']:.2f}"
            })
        
        # Print comparison table
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False))
            
            # Training recommendations
            print(f"\n📋 TRAINING RECOMMENDATIONS:")
            for i, row in df.iterrows():
                file_name = row['File']
                tu_count = row['Translation Units']
                empty_rate = float(row['Avg Empty Rate (%)'])
                duplicate_rate = float(row['Avg Duplicate Rate (%)'])
                
                print(f"\n  {file_name}:")
                
                # Training approach recommendation
                if tu_count >= 15000:
                    training_approach = "✅ Full Custom Training"
                elif tu_count >= 5000:
                    training_approach = "⚠️ Customization + Adaptation"
                else:
                    training_approach = "💡 Glossary-based Enhancement"
                
                print(f"    • Recommended approach: {training_approach}")
                
                # Quality assessment
                quality_issues = []
                if empty_rate > 10:
                    quality_issues.append(f"High empty rate ({empty_rate}%)")
                if duplicate_rate > 30:
                    quality_issues.append(f"High duplicate rate ({duplicate_rate}%)")
                
                if quality_issues:
                    print(f"    • ⚠️ Quality concerns: {', '.join(quality_issues)}")
                else:
                    print(f"    • ✅ Good data quality")
                
                # Data cleaning recommendations
                if empty_rate > 5 or duplicate_rate > 20:
                    print(f"    • 🔧 Recommended preprocessing:")
                    if empty_rate > 5:
                        print(f"      - Remove empty segments")
                    if duplicate_rate > 20:
                        print(f"      - Deduplicate content")

# Usage example
def main():
    # Replace with your actual file paths
    tmx_files = [
        'path/to/file1.tmx',  # 3K units
        'path/to/file2.tmx',  # 8K units
        'path/to/file3.tmx'   # 18K units
    ]
    
    # Create analyzer and run analysis
    analyzer = TMXAnalyzer(tmx_files)
    results = analyzer.analyze_all_files()
    
    # Save results to file (optional)
    # You can add code here to save results to JSON or CSV
    
    return results

if __name__ == "__main__":
    # Example usage:
    print("TMX Analysis Tool")
    print("Replace the file paths in the main() function with your actual TMX file paths")
    print("\nTo run analysis:")
    print("1. Update tmx_files list with your file paths")
    print("2. Run: python tmx_analyzer.py")
    print("3. Or call: analyzer = TMXAnalyzer(your_file_paths); results = analyzer.analyze_all_files()")
    
    # Uncomment the following line after updating file paths
    # main()
