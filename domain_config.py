"""
Domain configuration management for the Document Chatbot Framework.

This module provides functionality to load and manage domain-specific configurations
for different types of document chatbots (HR, Legal, Technical, etc.).
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DomainConfig:
    """Domain configuration data class."""
    
    # Basic info
    name: str
    description: str
    domain: str
    
    # UI configuration
    ui_title: str
    ui_page_title: str
    ui_page_icon: str
    ui_sidebar_title: str
    ui_footer: str
    
    # Document configuration
    documents_folder: str
    documents_folder_display_name: str
    documents_file_types: List[str]
    documents_index_path: str
    
    # Query configuration
    query_placeholder: str
    query_help_text: str
    query_button_text: str
    
    # Prompt configuration
    system_prompt: str
    
    # Processing configuration
    chunk_size: int
    chunk_overlap: int
    default_k: int
    max_k: int
    
    # Messages
    messages: Dict[str, str]


class DomainConfigLoader:
    """Loads and manages domain configurations."""
    
    def __init__(self, domains_folder: str = "./domains"):
        """
        Initialize the domain config loader.
        
        Args:
            domains_folder: Path to the folder containing domain YAML files
        """
        self.domains_folder = Path(domains_folder)
        self._configs_cache: Dict[str, DomainConfig] = {}
    
    def list_available_domains(self) -> List[str]:
        """
        List all available domain configurations.
        
        Returns:
            List of domain names
        """
        if not self.domains_folder.exists():
            logger.warning(f"Domains folder not found: {self.domains_folder}")
            return []
        
        domains = []
        for file_path in self.domains_folder.glob("*.yaml"):
            domain_name = file_path.stem
            domains.append(domain_name)
        
        return sorted(domains)
    
    def load_domain_config(self, domain: str) -> DomainConfig:
        """
        Load configuration for a specific domain.
        
        Args:
            domain: Domain name (e.g., 'hr', 'legal', 'technical')
            
        Returns:
            DomainConfig object
            
        Raises:
            FileNotFoundError: If domain config file doesn't exist
            ValueError: If config is invalid
        """
        # Check cache first
        if domain in self._configs_cache:
            return self._configs_cache[domain]
        
        config_file = self.domains_folder / f"{domain}.yaml"
        
        if not config_file.exists():
            available = self.list_available_domains()
            raise FileNotFoundError(
                f"Domain config '{domain}' not found. Available domains: {available}"
            )
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Parse and validate configuration
            domain_config = self._parse_config(config_data)
            
            # Cache the config
            self._configs_cache[domain] = domain_config
            
            logger.info(f"Loaded domain config: {domain}")
            return domain_config
            
        except Exception as e:
            raise ValueError(f"Error loading domain config '{domain}': {str(e)}")
    
    def _parse_config(self, config_data: Dict[str, Any]) -> DomainConfig:
        """
        Parse and validate configuration data.
        
        Args:
            config_data: Raw configuration dictionary
            
        Returns:
            DomainConfig object
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            return DomainConfig(
                # Basic info
                name=config_data['name'],
                description=config_data['description'],
                domain=config_data['domain'],
                
                # UI configuration
                ui_title=config_data['ui']['title'],
                ui_page_title=config_data['ui']['page_title'],
                ui_page_icon=config_data['ui']['page_icon'],
                ui_sidebar_title=config_data['ui']['sidebar_title'],
                ui_footer=config_data['ui']['footer'],
                
                # Document configuration
                documents_folder=config_data['documents']['folder'],
                documents_folder_display_name=config_data['documents']['folder_display_name'],
                documents_file_types=config_data['documents']['file_types'],
                documents_index_path=config_data['documents']['index_path'],
                
                # Query configuration
                query_placeholder=config_data['query']['placeholder'],
                query_help_text=config_data['query']['help_text'],
                query_button_text=config_data['query']['button_text'],
                
                # Prompt configuration
                system_prompt=config_data['prompts']['system_prompt'],
                
                # Processing configuration
                chunk_size=config_data['processing']['chunk_size'],
                chunk_overlap=config_data['processing']['chunk_overlap'],
                default_k=config_data['processing']['default_k'],
                max_k=config_data['processing']['max_k'],
                
                # Messages
                messages=config_data['messages'],
            )
            
        except KeyError as e:
            raise ValueError(f"Missing required configuration key: {e}")
        except Exception as e:
            raise ValueError(f"Invalid configuration format: {e}")
    
    def get_domain_from_env(self) -> str:
        """
        Get domain from environment variable or default to 'hr'.
        
        Returns:
            Domain name
        """
        return os.getenv('CHATBOT_DOMAIN', 'hr')
    
    def validate_domain_setup(self, domain_config: DomainConfig) -> List[str]:
        """
        Validate that the domain is properly set up.
        
        Args:
            domain_config: Domain configuration to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Check if documents folder exists
        if not os.path.exists(domain_config.documents_folder):
            issues.append(f"Documents folder not found: {domain_config.documents_folder}")
        
        # Check if documents folder has files
        if os.path.exists(domain_config.documents_folder):
            files = [
                f for f in os.listdir(domain_config.documents_folder)
                if any(f.lower().endswith(ext) for ext in domain_config.documents_file_types)
            ]
            if not files:
                issues.append(
                    f"No {', '.join(domain_config.documents_file_types)} files found in "
                    f"{domain_config.documents_folder}"
                )
        
        return issues


# Global domain config loader instance
_domain_loader: Optional[DomainConfigLoader] = None


def get_domain_loader() -> DomainConfigLoader:
    """Get the global domain config loader instance."""
    global _domain_loader
    if _domain_loader is None:
        _domain_loader = DomainConfigLoader()
    return _domain_loader


def load_current_domain_config() -> DomainConfig:
    """
    Load the current domain configuration based on environment.
    
    Returns:
        DomainConfig for the current domain
    """
    loader = get_domain_loader()
    domain = loader.get_domain_from_env()
    return loader.load_domain_config(domain)


def get_available_domains() -> List[str]:
    """Get list of available domain configurations."""
    loader = get_domain_loader()
    return loader.list_available_domains()


# Convenience functions for backward compatibility
def get_domain_config(domain: str = None) -> DomainConfig:
    """
    Get domain configuration.
    
    Args:
        domain: Domain name (defaults to environment or 'hr')
        
    Returns:
        DomainConfig object
    """
    loader = get_domain_loader()
    if domain is None:
        domain = loader.get_domain_from_env()
    return loader.load_domain_config(domain)