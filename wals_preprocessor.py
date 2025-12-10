# wals_preprocessor.py

import pandas as pd
import numpy as np
from pathlib import Path
import json

class WALSPreprocessor:
    def __init__(self, wals_dir: str):
        """
        Args:
            wals_dir: Path to extracted WALS dataset directory
        """
        self.wals_dir = Path(wals_dir)
        self.languages = pd.read_csv(self.wals_dir / 'cldf' / 'languages.csv')
        self.values = pd.read_csv(self.wals_dir / 'cldf' / 'values.csv')
        self.parameters = pd.read_csv(self.wals_dir / 'cldf' / 'parameters.csv')
        
    def extract_features(self, 
                        feature_ids: list = None,
                        min_features_per_lang: int = 10,
                        normalize: bool = True):
        """
        Extract and normalize WALS features for each language.
        
        Args:
            feature_ids: List of WALS feature IDs to use (e.g., ['1A', '2A', '3A'])
                        If None, uses all available features
            min_features_per_lang: Minimum features required for a language
            normalize: Whether to normalize categorical features
        """
        
        # Pivot values to get language x feature matrix
        feature_matrix = self.values.pivot(
            index='Language_ID', 
            columns='Parameter_ID', 
            values='Value'
        )
        
        # Filter features if specified
        if feature_ids:
            available_features = [f for f in feature_ids if f in feature_matrix.columns]
            feature_matrix = feature_matrix[available_features]
            print(f"Using {len(available_features)} features out of {len(feature_ids)} requested")
        
        # Filter languages with minimum features
        feature_counts = feature_matrix.notna().sum(axis=1)
        valid_langs = feature_counts[feature_counts >= min_features_per_lang].index
        feature_matrix = feature_matrix.loc[valid_langs]
        
        print(f"Retained {len(feature_matrix)} languages with >= {min_features_per_lang} features")
        
        # Add language metadata
        lang_info = self.languages.set_index('ID')[['ISO639P3code', 'Name', 'Latitude', 'Longitude']]
        result_df = feature_matrix.join(lang_info)
        
        # Reorganize columns
        metadata_cols = ['ISO639P3code', 'Name', 'Latitude', 'Longitude']
        feature_cols = [col for col in result_df.columns if col not in metadata_cols]
        
        # Create final dataframe
        final_df = pd.DataFrame()
        final_df['iso_code'] = result_df['ISO639P3code']
        final_df['language'] = result_df['Name']
        final_df['latitude'] = result_df['Latitude']
        final_df['longitude'] = result_df['Longitude']
        
        # Process features
        for feat_id in feature_cols:
            if normalize:
                # Normalize categorical features to 0-1 range
                final_df[feat_id] = self._normalize_feature(result_df[feat_id])
            else:
                final_df[feat_id] = result_df[feat_id]
        
        return final_df
    
    def _normalize_feature(self, feature_series):
        """
        Normalize a WALS feature to 0-1 range.
        Handles categorical values (1, 2, 3, etc.)
        """
        # Get unique non-null values
        unique_vals = feature_series.dropna().unique()
        
        if len(unique_vals) <= 1:
            return feature_series
        
        # Map categorical values to normalized range
        min_val = min(unique_vals)
        max_val = max(unique_vals)
        
        if max_val == min_val:
            return feature_series
        
        # Normalize to 0-1
        normalized = (feature_series - min_val) / (max_val - min_val)
        
        return normalized
    
    def get_feature_descriptions(self, feature_ids: list = None):
        """Get descriptions for features."""
        if feature_ids:
            params = self.parameters[self.parameters['ID'].isin(feature_ids)]
        else:
            params = self.parameters
        
        return dict(zip(params['ID'], params['Name']))

# Usage function
def prepare_wals_data(wals_dir: str, output_file: str = 'wals_features.csv'):
    """
    Main function to prepare WALS data for the typology module.
    """
    preprocessor = WALSPreprocessor(wals_dir)
    
    # Common typological features to use (you can modify this list)
    # These are some of the most informative WALS features
    important_features = [
        '81A',  # Order of Subject, Object and Verb
        '82A',  # Order of Subject and Verb
        '83A',  # Order of Object and Verb
        '85A',  # Order of Adposition and Noun Phrase
        '86A',  # Order of Genitive and Noun
        '87A',  # Order of Adjective and Noun
        '88A',  # Order of Demonstrative and Noun
        '89A',  # Order of Numeral and Noun
        '90A',  # Order of Relative Clause and Noun
        '95A',  # Relationship between the Order of Object and Verb
        '20A',  # Fusion of Selected Inflectional Formatives
        '21A',  # Exponence of Selected Inflectional Formatives
        '22A',  # Inflectional Synthesis of the Verb
        '26A',  # Prefixing vs. Suffixing in Inflectional Morphology
        '49A',  # Number of Cases
        '30A',  # Number of Genders
        '33A',  # Coding of Nominal Plurality
        '37A',  # Definite Articles
        '38A',  # Indefinite Articles
        '112A', # Negative Morphemes
        '143A', # Order of Negative Morpheme and Verb
        '144A', # Position of Negative Word
        '69A',  # Position of Tense-Aspect Affixes
        '1A',   # Consonant Inventories
        '2A',   # Vowel Quality Inventories
        '3A',   # Consonant-Vowel Ratio
        '12A',  # Syllable Structure
        '13A',  # Tone
    ]
    
    # Extract features
    df = preprocessor.extract_features(
        feature_ids=important_features,
        min_features_per_lang=15,  # Require at least 15 features
        normalize=True
    )
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(df)} languages to {output_file}")
    
    # Print statistics
    print(f"\nFeature coverage:")
    feature_cols = [col for col in df.columns if col not in ['iso_code', 'language', 'latitude', 'longitude']]
    for feat in feature_cols[:10]:  # Show first 10
        coverage = df[feat].notna().mean() * 100
        print(f"  {feat}: {coverage:.1f}% coverage")
    
    # Get feature descriptions
    descriptions = preprocessor.get_feature_descriptions(important_features)
    with open('wals_feature_descriptions.json', 'w') as f:
        json.dump(descriptions, f, indent=2)
    print(f"\nSaved feature descriptions to wals_feature_descriptions.json")
    
    return df

