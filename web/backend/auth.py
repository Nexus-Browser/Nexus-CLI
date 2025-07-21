"""
Auth0 Authentication Module for Nexus CLI Web Interface
Handles JWT token validation and user management
"""

import os
import json
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import requests
from functools import wraps

try:
    from jose import jwt
    from jose.exceptions import JWTError
    from authlib.integrations.starlette_client import OAuth
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("Warning: JWT dependencies not available. Auth will be disabled.")

class Auth0Manager:
    """Auth0 authentication and user management"""
    
    def __init__(self):
        # Auth0 Configuration
        self.domain = os.getenv('AUTH0_DOMAIN', 'your-domain.us.auth0.com')
        self.client_id = os.getenv('AUTH0_CLIENT_ID', 'your-client-id')
        self.client_secret = os.getenv('AUTH0_CLIENT_SECRET', 'your-client-secret')
        self.audience = os.getenv('AUTH0_AUDIENCE', f'https://{self.domain}/api/v2/')
        self.algorithm = 'RS256'
        
        # JWT Configuration
        self.jwks_url = f'https://{self.domain}/.well-known/jwks.json'
        self.issuer = f'https://{self.domain}/'
        
        # Cache for JWKS
        self._jwks_cache = None
        self._jwks_cache_expiry = None
        
        # In-memory user storage (fallback)
        self.users = {}
        
        print(f"Auth0 Manager initialized for domain: {self.domain}")
        
    def get_jwks(self) -> Dict[str, Any]:
        """Get JSON Web Key Set from Auth0"""
        now = datetime.now()
        
        # Check if cached JWKS is still valid (cache for 1 hour)
        if (self._jwks_cache and self._jwks_cache_expiry and 
            now < self._jwks_cache_expiry):
            return self._jwks_cache
            
        try:
            response = requests.get(self.jwks_url, timeout=10)
            response.raise_for_status()
            jwks = response.json()
            
            # Cache the JWKS
            self._jwks_cache = jwks
            self._jwks_cache_expiry = now + timedelta(hours=1)
            
            return jwks
        except Exception as e:
            print(f"Failed to fetch JWKS: {e}")
            # Return cached version if available
            if self._jwks_cache:
                return self._jwks_cache
            return {"keys": []}
    
    def get_rsa_key(self, token: str) -> Optional[Dict[str, Any]]:
        """Extract RSA key from JWKS for token validation"""
        if not JWT_AVAILABLE:
            return None
            
        try:
            # Decode token header without verification to get kid
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get('kid')
            
            if not kid:
                return None
                
            # Get JWKS and find matching key
            jwks = self.get_jwks()
            for key in jwks.get('keys', []):
                if key.get('kid') == kid:
                    return key
                    
            return None
        except Exception as e:
            print(f"Error getting RSA key: {e}")
            return None
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        if not JWT_AVAILABLE:
            print("JWT not available, skipping token verification")
            return {"sub": "demo-user", "email": "demo@example.com", "name": "Demo User"}
            
        try:
            # Get RSA key for token
            rsa_key = self.get_rsa_key(token)
            if not rsa_key:
                print("No matching RSA key found")
                return None
                
            # Construct the key for verification
            key = {
                'kty': rsa_key['kty'],
                'kid': rsa_key['kid'],
                'use': rsa_key['use'],
                'n': rsa_key['n'],
                'e': rsa_key['e']
            }
            
            # Verify and decode the token
            payload = jwt.decode(
                token,
                key,
                algorithms=[self.algorithm],
                audience=self.audience,
                issuer=self.issuer
            )
            
            # Store user info in cache
            user_id = payload.get('sub')
            if user_id:
                self.users[user_id] = {
                    'id': user_id,
                    'email': payload.get('email'),
                    'name': payload.get('name'),
                    'picture': payload.get('picture'),
                    'last_seen': datetime.now(),
                    'permissions': payload.get('permissions', [])
                }
            
            return payload
            
        except JWTError as e:
            print(f"JWT verification failed: {e}")
            return None
        except Exception as e:
            print(f"Token verification error: {e}")
            return None
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information from cache or Auth0"""
        # Check local cache first
        if user_id in self.users:
            return self.users[user_id]
            
        # For demo purposes, return a mock user
        return {
            'id': user_id,
            'email': f"{user_id}@example.com",
            'name': "Nexus User",
            'picture': None,
            'last_seen': datetime.now(),
            'permissions': ['read', 'write', 'execute']
        }
    
    def create_session_token(self, user_id: str, session_id: str) -> str:
        """Create a simple session token for internal use"""
        import hashlib
        import time
        
        # Create a simple hash-based token
        data = f"{user_id}:{session_id}:{int(time.time())}"
        token = hashlib.sha256(data.encode()).hexdigest()
        return token
    
    def is_authenticated(self, token: str) -> bool:
        """Check if token is valid"""
        payload = self.verify_token(token)
        return payload is not None
    
    def get_auth_url(self, redirect_uri: str, state: str = None) -> str:
        """Generate Auth0 authorization URL"""
        import urllib.parse
        
        params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': redirect_uri,
            'scope': 'openid profile email',
            'audience': self.audience
        }
        
        if state:
            params['state'] = state
            
        query_string = urllib.parse.urlencode(params)
        return f"https://{self.domain}/authorize?{query_string}"
    
    def exchange_code_for_token(self, code: str, redirect_uri: str) -> Optional[Dict[str, Any]]:
        """Exchange authorization code for access token"""
        try:
            token_url = f"https://{self.domain}/oauth/token"
            
            payload = {
                'grant_type': 'authorization_code',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'code': code,
                'redirect_uri': redirect_uri
            }
            
            headers = {'Content-Type': 'application/json'}
            
            response = requests.post(token_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"Token exchange failed: {e}")
            return None

# Global auth manager instance
auth_manager = Auth0Manager()
