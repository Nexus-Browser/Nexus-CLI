import logging
import re
from typing import Optional

class NexusModel:
    """Enhanced rule-based Nexus AI model for code generation and conversation."""
    
    def __init__(self, model_path: str = "./model/nexus_model"):
        """Initialize the Nexus model."""
        logging.info("Using enhanced rule-based Nexus model")
        
        # Enhanced code templates with more specific patterns
        self.code_templates = {
            "python": {
                "calculator": """class Calculator:
    \"\"\"
    Simple calculator class with basic arithmetic operations
    \"\"\"
    
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def power(self, a, b):
        return a ** b
    
    def square_root(self, a):
        if a < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return a ** 0.5

# Example usage
calc = Calculator()
print(calc.add(5, 3))      # 8
print(calc.multiply(4, 7))  # 28
print(calc.power(2, 8))     # 256""",
                
                "web_server": """from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Hello, World!", "status": "running"})

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"data": "Some data", "timestamp": "2024-01-01"})

@app.route('/api/data', methods=['POST'])
def post_data():
    data = request.get_json()
    return jsonify({"received": data, "status": "success"})

@app.route('/api/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    operation = data.get('operation')
    a = data.get('a', 0)
    b = data.get('b', 0)
    
    if operation == 'add':
        result = a + b
    elif operation == 'multiply':
        result = a * b
    else:
        return jsonify({"error": "Unknown operation"}), 400
    
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)""",
                
                "file_handler": """import os
import json
import csv
from pathlib import Path

class FileHandler:
    \"\"\"
    Utility class for file operations
    \"\"\"
    
    @staticmethod
    def read_file(filename):
        \"\"\"Read a text file and return its contents.\"\"\"
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None
        except Exception as e:
            return f"Error reading file: {e}"
    
    @staticmethod
    def write_file(filename, content):
        \"\"\"Write content to a file.\"\"\"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            return f"Error writing file: {e}"
    
    @staticmethod
    def append_file(filename, content):
        \"\"\"Append content to a file.\"\"\"
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            return f"Error appending to file: {e}"
    
    @staticmethod
    def read_json(filename):
        \"\"\"Read a JSON file and return parsed data.\"\"\"
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return f"Error reading JSON: {e}"
    
    @staticmethod
    def write_json(filename, data):
        \"\"\"Write data to a JSON file.\"\"\"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            return f"Error writing JSON: {e}"
    
    @staticmethod
    def list_files(directory="."):
        \"\"\"List all files in a directory.\"\"\"
        try:
            files = []
            for item in os.listdir(directory):
                path = os.path.join(directory, item)
                if os.path.isfile(path):
                    files.append(item)
            return files
        except Exception as e:
            return f"Error listing files: {e}"

# Example usage
handler = FileHandler()
content = handler.read_file("example.txt")
handler.write_file("output.txt", "Hello, World!")""",
                
                "data_structures": """# Common data structures and operations in Python

# Lists - Mutable, ordered collections
my_list = [1, 2, 3, 4, 5]
my_list.append(6)           # Add to end
my_list.extend([7, 8])      # Add multiple items
my_list.insert(0, 0)        # Insert at specific position
removed = my_list.pop()      # Remove and return last item
my_list.remove(3)           # Remove first occurrence of value
my_list.sort()              # Sort in place
sorted_list = sorted(my_list)  # Return new sorted list

# Dictionaries - Key-value pairs
my_dict = {"name": "John", "age": 30, "city": "New York"}
my_dict["country"] = "USA"  # Add new key-value pair
my_dict.update({"email": "john@example.com"})  # Update with another dict
value = my_dict.get("phone", "Not found")  # Get with default
del my_dict["age"]          # Remove key-value pair
keys = list(my_dict.keys()) # Get all keys
values = list(my_dict.values())  # Get all values

# Sets - Unordered, unique elements
my_set = {1, 2, 3, 4, 5}
my_set.add(6)               # Add element
my_set.remove(1)            # Remove element (raises error if not found)
my_set.discard(10)          # Remove element (no error if not found)
set2 = {4, 5, 6, 7}
union = my_set | set2       # Union
intersection = my_set & set2  # Intersection
difference = my_set - set2  # Difference

# Tuples - Immutable, ordered
my_tuple = (1, 2, 3, 4, 5)
# my_tuple[0] = 10  # This would raise an error
tuple_length = len(my_tuple)

# Stack implementation using list
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Queue implementation
from collections import deque
queue = deque()
queue.append(1)  # Enqueue
queue.append(2)
queue.append(3)
first_item = queue.popleft()  # Dequeue""",
                
                "sorting_algorithms": """# Common sorting algorithms in Python

def bubble_sort(arr):
    \"\"\"Bubble sort algorithm.\"\"\"
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def selection_sort(arr):
    \"\"\"Selection sort algorithm.\"\"\"
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def insertion_sort(arr):
    \"\"\"Insertion sort algorithm.\"\"\"
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def merge_sort(arr):
    \"\"\"Merge sort algorithm.\"\"\"
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    \"\"\"Merge two sorted arrays.\"\"\"
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr):
    \"\"\"Quick sort algorithm.\"\"\"
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# Example usage
numbers = [64, 34, 25, 12, 22, 11, 90]
print(f"Original: {numbers}")
print(f"Bubble sort: {bubble_sort(numbers.copy())}")
print(f"Selection sort: {selection_sort(numbers.copy())}")
print(f"Insertion sort: {insertion_sort(numbers.copy())}")
print(f"Merge sort: {merge_sort(numbers.copy())}")
print(f"Quick sort: {quick_sort(numbers.copy())}")""",
                
                "searching_algorithms": """# Common searching algorithms in Python

def linear_search(arr, target):
    \"\"\"Linear search - O(n) time complexity.\"\"\"
    for i, element in enumerate(arr):
        if element == target:
            return i
    return -1

def binary_search(arr, target):
    \"\"\"Binary search - O(log n) time complexity. Requires sorted array.\"\"\"
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def binary_search_recursive(arr, target, left=0, right=None):
    \"\"\"Recursive binary search.\"\"\"
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

def find_min(arr):
    \"\"\"Find minimum element in array.\"\"\"
    if not arr:
        return None
    return min(arr)

def find_max(arr):
    \"\"\"Find maximum element in array.\"\"\"
    if not arr:
        return None
    return max(arr)

def find_duplicates(arr):
    \"\"\"Find duplicate elements in array.\"\"\"
    seen = set()
    duplicates = set()
    
    for element in arr:
        if element in seen:
            duplicates.add(element)
        else:
            seen.add(element)
    
    return list(duplicates)

# Example usage
numbers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
target = 7

print(f"Array: {numbers}")
print(f"Linear search for {target}: {linear_search(numbers, target)}")
print(f"Binary search for {target}: {binary_search(numbers, target)}")
print(f"Recursive binary search for {target}: {binary_search_recursive(numbers, target)}")
print(f"Minimum: {find_min(numbers)}")
print(f"Maximum: {find_max(numbers)}")

# Test with duplicates
duplicate_array = [1, 2, 3, 4, 2, 5, 6, 3, 7, 8, 1]
print(f"Duplicates in {duplicate_array}: {find_duplicates(duplicate_array)}")""",
                
                "web_scraping": """# Web scraping with Python

import requests
from bs4 import BeautifulSoup
import json
import csv
from urllib.parse import urljoin, urlparse

class WebScraper:
    \"\"\"
    Simple web scraper class
    \"\"\"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_page(self, url):
        \"\"\"Get HTML content from URL.\"\"\"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def parse_html(self, html):
        \"\"\"Parse HTML with BeautifulSoup.\"\"\"
        if html:
            return BeautifulSoup(html, 'html.parser')
        return None
    
    def extract_links(self, soup, base_url):
        \"\"\"Extract all links from page.\"\"\"
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            links.append({
                'text': link.get_text(strip=True),
                'url': full_url
            })
        return links
    
    def extract_text(self, soup, selector=None):
        \"\"\"Extract text from page or specific elements.\"\"\"
        if selector:
            elements = soup.select(selector)
            return [elem.get_text(strip=True) for elem in elements]
        else:
            return soup.get_text(strip=True)
    
    def save_to_json(self, data, filename):
        \"\"\"Save data to JSON file.\"\"\"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def save_to_csv(self, data, filename):
        \"\"\"Save data to CSV file.\"\"\"
        if data and isinstance(data, list):
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

# Example usage
scraper = WebScraper()

# Scrape a simple website
url = "https://httpbin.org/html"
html = scraper.get_page(url)
if html:
    soup = scraper.parse_html(html)
    if soup:
        title = soup.find('title')
        if title:
            print(f"Page title: {title.get_text()}")
        
        # Extract all text
        text = scraper.extract_text(soup)
        print(f"Page text length: {len(text)} characters")

# Note: Install required packages with: pip install requests beautifulsoup4""",
                
                "database_operations": """# Database operations with SQLite

import sqlite3
import json
from datetime import datetime

class DatabaseManager:
    \"\"\"
    Simple database manager using SQLite
    \"\"\"
    
    def __init__(self, db_name="nexus.db"):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
    
    def connect(self):
        \"\"\"Connect to database.\"\"\"
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.cursor = self.conn.cursor()
            return True
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            return False
    
    def disconnect(self):
        \"\"\"Disconnect from database.\"\"\"
        if self.conn:
            self.conn.close()
    
    def create_table(self, table_name, columns):
        \"\"\"Create a new table.\"\"\"
        if not self.connect():
            return False
        
        try:
            column_definitions = ', '.join(columns)
            query = f\"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions})\"
            self.cursor.execute(query)
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error creating table: {e}")
            return False
        finally:
            self.disconnect()
    
    def insert_data(self, table_name, data):
        \"\"\"Insert data into table.\"\"\"
        if not self.connect():
            return False
        
        try:
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])
            query = f\"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})\"
            self.cursor.execute(query, list(data.values()))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error inserting data: {e}")
            return False
        finally:
            self.disconnect()
    
    def select_data(self, table_name, columns=None, where=None):
        \"\"\"Select data from table.\"\"\"
        if not self.connect():
            return []
        
        try:
            cols = '*' if columns is None else ', '.join(columns)
            query = f\"SELECT {cols} FROM {table_name}\"
            
            if where:
                query += f\" WHERE {where}\"
            
            self.cursor.execute(query)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error selecting data: {e}")
            return []
        finally:
            self.disconnect()
    
    def update_data(self, table_name, data, where):
        \"\"\"Update data in table.\"\"\"
        if not self.connect():
            return False
        
        try:
            set_clause = ', '.join([f\"{k} = ?\" for k in data.keys()])
            query = f\"UPDATE {table_name} SET {set_clause} WHERE {where}\"
            self.cursor.execute(query, list(data.values()))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating data: {e}")
            return False
        finally:
            self.disconnect()
    
    def delete_data(self, table_name, where):
        \"\"\"Delete data from table.\"\"\"
        if not self.connect():
            return False
        
        try:
            query = f\"DELETE FROM {table_name} WHERE {where}\"
            self.cursor.execute(query)
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error deleting data: {e}")
            return False
        finally:
            self.disconnect()

# Example usage
db = DatabaseManager()

# Create a users table
columns = [
    \"id INTEGER PRIMARY KEY AUTOINCREMENT\",
    \"name TEXT NOT NULL\",
    \"email TEXT UNIQUE\",
    \"created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\"
]
db.create_table(\"users\", columns)

# Insert a user
user_data = {
    \"name\": \"John Doe\",
    \"email\": \"john@example.com\"
}
db.insert_data(\"users\", user_data)

# Select all users
users = db.select_data(\"users\")
print(f\"Users: {users}\")

# Update user
update_data = {\"name\": \"Jane Doe\"}
db.update_data(\"users\", update_data, \"email = 'john@example.com'\")

# Select updated user
updated_users = db.select_data(\"users\", where=\"email = 'john@example.com'\")
print(f\"Updated user: {updated_users}\")""",
                
                "api_client": """# API client for making HTTP requests

import requests
import json
from typing import Dict, Any, Optional

class APIClient:
    \"\"\"
    Simple API client for making HTTP requests
    \"\"\"
    
    def __init__(self, base_url: str = "", headers: Dict[str, str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Set default headers
        default_headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Nexus-CLI/1.0'
        }
        
        if headers:
            default_headers.update(headers)
        
        self.session.headers.update(default_headers)
    
    def get(self, endpoint: str = "", params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        \"\"\"Make a GET request.\"\"\"
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"GET request error: {e}")
            return None
    
    def post(self, endpoint: str = "", data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        \"\"\"Make a POST request.\"\"\"
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"POST request error: {e}")
            return None
    
    def put(self, endpoint: str = "", data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        \"\"\"Make a PUT request.\"\"\"
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url
            response = self.session.put(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"PUT request error: {e}")
            return None
    
    def delete(self, endpoint: str = "") -> bool:
        \"\"\"Make a DELETE request.\"\"\"
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url
            response = self.session.delete(url)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"DELETE request error: {e}")
            return False
    
    def download_file(self, url: str, filename: str) -> bool:
        \"\"\"Download a file from URL.\"\"\"
        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except requests.RequestException as e:
            print(f"Download error: {e}")
            return False

# Example usage
# JSONPlaceholder API
api = APIClient("https://jsonplaceholder.typicode.com")

# Get all posts
posts = api.get("posts")
if posts:
    print(f"Found {len(posts)} posts")

# Get specific post
post = api.get("posts/1")
if post:
    print(f"Post title: {post.get('title')}")

# Create new post
new_post = {
    "title": "My New Post",
    "body": "This is the content of my new post",
    "userId": 1
}
created_post = api.post("posts", new_post)
if created_post:
    print(f"Created post with ID: {created_post.get('id')}")

# Update post
update_data = {"title": "Updated Title"}
updated_post = api.put("posts/1", update_data)
if updated_post:
    print(f"Updated post: {updated_post.get('title')}")

# Delete post
success = api.delete("posts/1")
if success:
    print("Post deleted successfully")

# GitHub API example
github_api = APIClient("https://api.github.com", headers={
    "Accept": "application/vnd.github.v3+json"
})

# Get user info (public API, no auth required)
user = github_api.get("users/octocat")
if user:
    print(f"GitHub user: {user.get('login')} - {user.get('name')}")"""
            }
        }
        
        # Common conversation responses
        self.conversation_responses = {
            "greeting": [
                "Hello! I'm Nexus, your AI coding assistant. How can I help you today?",
                "Hi there! I'm here to help with your coding tasks. What would you like to work on?",
                "Greetings! I'm Nexus, ready to assist with programming and development tasks."
            ],
            "help": [
                "I can help you with code generation, file operations, code analysis, and more! Just let me know what you need.",
                "I'm here to help with programming tasks. I can generate code, analyze existing code, help with debugging, and answer programming questions.",
                "I specialize in coding assistance. I can create functions, classes, web servers, handle files, and explain programming concepts."
            ],
            "variable": [
                "A variable is a named storage location in memory that holds data. In Python, you create variables by assigning values to names. For example: `x = 5` creates a variable named 'x' with the value 5.",
                "Variables are containers for storing data values. They have names and can hold different types of data like numbers, strings, lists, etc. You can change their values during program execution.",
                "Think of variables as labeled boxes that store information. You can put data in them, change the data, and reference them by their names throughout your program."
            ],
            "function": [
                "A function is a reusable block of code that performs a specific task. Functions help organize code and avoid repetition. They can take inputs (parameters) and return outputs.",
                "Functions are like mini-programs within your program. They group related code together and can be called multiple times. They make code more readable and maintainable.",
                "Functions are defined using the 'def' keyword in Python. They can accept parameters, perform operations, and return results. This promotes code reuse and modularity."
            ],
            "class": [
                "A class is a blueprint for creating objects. It defines the properties and methods that objects of that class will have. Classes are fundamental to object-oriented programming.",
                "Classes are templates for creating objects with shared behavior and data. They encapsulate data and methods that operate on that data, promoting code organization.",
                "Think of a class as a cookie cutter - it defines the shape and properties that all objects created from it will have. Classes help model real-world entities in code."
            ],
            "web_server": [
                "A web server is a program that handles HTTP requests and serves web pages or data. Popular Python frameworks include Flask, Django, and FastAPI.",
                "Web servers listen for incoming requests and respond with HTML pages, JSON data, or other content. They're essential for web applications and APIs.",
                "Web servers run continuously, waiting for client requests. They can serve static files, process form data, handle authentication, and interact with databases."
            ],
            "debugging": [
                "Debugging is the process of finding and fixing errors in code. Use print statements, logging, and debugging tools to identify issues.",
                "Common debugging techniques include: adding print statements, using a debugger, checking error messages, and testing code step by step.",
                "When debugging, start by reproducing the error, then trace through the code to find where things go wrong. Use tools like pdb in Python."
            ]
        }
    
    def generate_response(self, prompt: str, max_length: int = 128, temperature: float = 0.7) -> str:
        """
        Generate a response to a prompt using rule-based logic.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated response (ignored in rule-based)
            temperature: Sampling temperature (ignored in rule-based)
            
        Returns:
            Generated response
        """
        prompt_lower = prompt.lower()
        
        # Handle greetings
        if any(word in prompt_lower for word in ["hello", "hi", "hey", "greetings"]):
            import random
            return random.choice(self.conversation_responses["greeting"])
        
        # Handle help requests
        if any(word in prompt_lower for word in ["help", "what can you do", "capabilities"]):
            import random
            return random.choice(self.conversation_responses["help"])
        
        # Handle specific questions
        if "variable" in prompt_lower:
            import random
            return random.choice(self.conversation_responses["variable"])
        
        if "function" in prompt_lower:
            import random
            return random.choice(self.conversation_responses["function"])
        
        if "class" in prompt_lower:
            import random
            return random.choice(self.conversation_responses["class"])
        
        if "web server" in prompt_lower or "webserver" in prompt_lower:
            import random
            return random.choice(self.conversation_responses["web_server"])
        
        if "debug" in prompt_lower or "error" in prompt_lower:
            import random
            return random.choice(self.conversation_responses["debugging"])
        
        # Default response
        return "I understand your question. I'm here to help with programming tasks. Could you please provide more details about what you'd like me to help you with?"
    
    def generate_code(self, instruction: str, language: str = "python") -> str:
        """
        Generate code based on a natural language instruction.
        
        Args:
            instruction: Natural language description of what code to generate
            language: Programming language for the code
            
        Returns:
            Generated code
        """
        instruction_lower = instruction.lower()
        
        # Handle specific code generation requests with better pattern matching
        if "calculator" in instruction_lower:
            return self.code_templates["python"]["calculator"]
        
        if "web server" in instruction_lower or "webserver" in instruction_lower or "flask" in instruction_lower:
            return self.code_templates["python"]["web_server"]
        
        if "file" in instruction_lower and ("read" in instruction_lower or "write" in instruction_lower or "handler" in instruction_lower):
            return self.code_templates["python"]["file_handler"]
        
        if "data structure" in instruction_lower or "list" in instruction_lower or "dictionary" in instruction_lower or "set" in instruction_lower or "tuple" in instruction_lower:
            return self.code_templates["python"]["data_structures"]
        
        if "sort" in instruction_lower or "sorting" in instruction_lower:
            return self.code_templates["python"]["sorting_algorithms"]
        
        if "search" in instruction_lower or "find" in instruction_lower:
            return self.code_templates["python"]["searching_algorithms"]
        
        if "scrape" in instruction_lower or "web scraping" in instruction_lower or "beautifulsoup" in instruction_lower:
            return self.code_templates["python"]["web_scraping"]
        
        if "database" in instruction_lower or "sqlite" in instruction_lower or "sql" in instruction_lower:
            return self.code_templates["python"]["database_operations"]
        
        if "api" in instruction_lower or "http" in instruction_lower or "request" in instruction_lower:
            return self.code_templates["python"]["api_client"]
        
        # Handle function requests with better pattern matching
        if "function" in instruction_lower or "def" in instruction_lower:
            words = instruction.split()
            function_name = "my_function"
            
            # Try to find a meaningful function name
            for i, word in enumerate(words):
                if word in ["function", "def", "create", "write", "make"] and i + 1 < len(words):
                    name_parts = words[i+1:i+4]
                    function_name = "_".join(name_parts).lower()
                    function_name = re.sub(r'[^a-zA-Z0-9_]', '', function_name)
                    break
            
            # Generate appropriate function based on instruction
            if "add" in instruction_lower or "sum" in instruction_lower:
                return f"""def {function_name}(a, b):
    \"\"\"
    Add two numbers together
    \"\"\"
    return a + b

# Example usage
result = {function_name}(5, 3)
print(result)  # 8"""
            
            elif "multiply" in instruction_lower:
                return f"""def {function_name}(a, b):
    \"\"\"
    Multiply two numbers
    \"\"\"
    return a * b

# Example usage
result = {function_name}(4, 7)
print(result)  # 28"""
            
            elif "check" in instruction_lower and "even" in instruction_lower:
                return f"""def {function_name}(number):
    \"\"\"
    Check if a number is even
    \"\"\"
    return number % 2 == 0

# Example usage
print({function_name}(4))  # True
print({function_name}(7))  # False"""
            
            elif "factorial" in instruction_lower:
                return f"""def {function_name}(n):
    \"\"\"
    Calculate factorial of a number
    \"\"\"
    if n <= 1:
        return 1
    return n * {function_name}(n - 1)

# Example usage
print({function_name}(5))  # 120"""
            
            elif "fibonacci" in instruction_lower:
                return f"""def {function_name}(n):
    \"\"\"
    Calculate the nth Fibonacci number
    \"\"\"
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return {function_name}(n-1) + {function_name}(n-2)

# Example usage
print({function_name}(10))  # 55"""
            
            elif "prime" in instruction_lower:
                return f"""def {function_name}(n):
    \"\"\"
    Check if a number is prime
    \"\"\"
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Example usage
print({function_name}(17))  # True
print({function_name}(24))  # False"""
            
            elif "palindrome" in instruction_lower:
                return f"""def {function_name}(s):
    \"\"\"
    Check if a string is a palindrome
    \"\"\"
    s = s.lower().replace(' ', '')
    return s == s[::-1]

# Example usage
print({function_name}("racecar"))  # True
print({function_name}("hello"))    # False"""
            
            elif "reverse" in instruction_lower:
                return f"""def {function_name}(s):
    \"\"\"
    Reverse a string
    \"\"\"
    return s[::-1]

# Example usage
print({function_name}("hello"))  # olleh"""
            
            elif "count" in instruction_lower and "vowel" in instruction_lower:
                return f"""def {function_name}(s):
    \"\"\"
    Count vowels in a string
    \"\"\"
    vowels = 'aeiouAEIOU'
    return sum(1 for char in s if char in vowels)

# Example usage
print({function_name}("hello world"))  # 3"""
            
            else:
                # Generic function template
                return f"""def {function_name}():
    \"\"\"
    {instruction}
    \"\"\"
    # TODO: Implement the function
    pass

# Example usage
result = {function_name}()
print(result)"""
        
        # Handle class requests
        if "class" in instruction_lower:
            class_name = "MyClass"
            
            # Try to extract class name
            for i, word in enumerate(words):
                if word in ["class", "create", "write", "make"] and i + 1 < len(words):
                    name_parts = words[i+1:i+3]
                    class_name = "".join(name_parts).title()
                    class_name = re.sub(r'[^a-zA-Z0-9]', '', class_name)
                    break
            
            return f"""class {class_name}:
    \"\"\"
    {instruction}
    \"\"\"
    
    def __init__(self):
        # Initialize the class
        pass
    
    def method1(self):
        # Add your methods here
        pass

# Example usage
obj = {class_name}()"""
        
        # Default code template
        return f"""# {instruction}

def main():
    \"\"\"
    Main function to implement {instruction}
    \"\"\"
    # TODO: Add your implementation here
    pass

if __name__ == "__main__":
    main()""" 