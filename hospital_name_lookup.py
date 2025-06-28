"""
Hospital Name Lookup Utility

This module provides functions to look up real hospital names from osph_id values
using the official California hospital financial data.

Usage:
    from hospital_name_lookup import get_hospital_name, load_hospital_mapping
    
    # Get a single hospital name
    name = get_hospital_name("106014132")
    print(name)  # "KAISER FOUNDATION HOSPITALS - FREMONT"
    
    # Load the full mapping
    mapping = load_hospital_mapping()
"""

import json
from pathlib import Path
from typing import Dict, Optional


def load_hospital_mapping() -> Dict[str, str]:
    """
    Load the complete hospital osph_id to name mapping.
    
    Returns:
        Dict[str, str]: Dictionary mapping osph_id to hospital legal name
        
    Raises:
        FileNotFoundError: If the mapping file doesn't exist
        json.JSONDecodeError: If the mapping file is corrupted
    """
    mapping_file = Path("hospital_osph_id_mapping.json")
    
    if not mapping_file.exists():
        raise FileNotFoundError(
            f"Hospital mapping file not found: {mapping_file}. "
            "Please run the hospital mapping extraction script first."
        )
    
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    
    return data['hospital_mapping']


def get_hospital_name(osph_id: str, default: Optional[str] = None) -> str:
    """
    Get the hospital name for a given osph_id.
    
    Args:
        osph_id (str): The hospital OSPH ID (can be string or numeric)
        default (Optional[str]): Default value if hospital not found.
                               If None, returns "Unknown Hospital (ID: {osph_id})"
    
    Returns:
        str: The hospital legal name or default value
        
    Examples:
        >>> get_hospital_name("106014132")
        'KAISER FOUNDATION HOSPITALS - FREMONT'
        
        >>> get_hospital_name("999999", "Hospital Not Found")
        'Hospital Not Found'
    """
    try:
        mapping = load_hospital_mapping()
        
        # Convert to string and handle different formats
        osph_id_str = str(osph_id).strip()
        
        # Try direct lookup
        if osph_id_str in mapping:
            return mapping[osph_id_str]
        
        # Try as integer (remove decimal points)
        try:
            osph_id_int = str(int(float(osph_id_str)))
            if osph_id_int in mapping:
                return mapping[osph_id_int]
        except (ValueError, TypeError):
            pass
        
        # Not found
        if default is not None:
            return default
        else:
            return f"Unknown Hospital (ID: {osph_id})"
            
    except Exception as e:
        if default is not None:
            return default
        else:
            return f"Error looking up hospital (ID: {osph_id}): {str(e)}"


def get_hospital_mapping_info() -> Dict:
    """
    Get information about the hospital mapping dataset.
    
    Returns:
        Dict: Information about the mapping including total count, creation date, etc.
    """
    mapping_file = Path("hospital_osph_id_mapping.json")
    
    if not mapping_file.exists():
        return {
            "error": "Hospital mapping file not found",
            "total_hospitals": 0,
            "creation_date": None
        }
    
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    
    return {
        "total_hospitals": data.get('total_hospitals', 0),
        "creation_date": data.get('creation_date'),
        "data_source": data.get('data_source'),
        "files_processed": len(data.get('files_processed', [])),
        "hospitals_with_name_changes": len(data.get('duplicates_info', {}))
    }


def search_hospitals(search_term: str, limit: int = 10) -> Dict[str, str]:
    """
    Search for hospitals by name containing the search term.
    
    Args:
        search_term (str): Term to search for in hospital names (case insensitive)
        limit (int): Maximum number of results to return
    
    Returns:
        Dict[str, str]: Dictionary of matching osph_id to hospital name
        
    Examples:
        >>> search_hospitals("KAISER")
        {'106014132': 'KAISER FOUNDATION HOSPITALS - FREMONT', ...}
        
        >>> search_hospitals("STANFORD")
        {'106014050': 'STANFORD HEALTH CARE TRI-VALLEY', ...}
    """
    try:
        mapping = load_hospital_mapping()
        search_lower = search_term.lower()
        
        matches = {}
        for osph_id, name in mapping.items():
            if search_lower in name.lower():
                matches[osph_id] = name
                if len(matches) >= limit:
                    break
        
        return matches
        
    except Exception:
        return {}


if __name__ == "__main__":
    # Demo usage
    print("=== Hospital Name Lookup Demo ===")
    
    # Show mapping info
    info = get_hospital_mapping_info()
    print(f"Total hospitals in mapping: {info['total_hospitals']}")
    print(f"Created: {info['creation_date']}")
    
    # Test lookups
    test_ids = ["106014132", "106014326", "106014050", "999999"]
    print(f"\n=== Sample Lookups ===")
    for osph_id in test_ids:
        name = get_hospital_name(osph_id)
        print(f"{osph_id}: {name}")
    
    # Search examples
    print(f"\n=== Search Examples ===")
    kaiser_hospitals = search_hospitals("KAISER", 5)
    print(f"Kaiser hospitals (first 5):")
    for osph_id, name in kaiser_hospitals.items():
        print(f"  {osph_id}: {name}")
    
    stanford_hospitals = search_hospitals("STANFORD")
    print(f"\nStanford hospitals:")
    for osph_id, name in stanford_hospitals.items():
        print(f"  {osph_id}: {name}") 