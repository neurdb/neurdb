#!/usr/bin/env python3
"""
MoLQO Test Server - Supports multiple optimization strategies
Provides different API endpoints to test different hint formats
"""

import json
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

class MoLQOHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Parse request path
            path = urlparse(self.path).path
            
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_json = json.loads(post_data.decode('utf-8'))
            
            # Get SQL query
            sql = request_json.get('sql', '').strip()
            
            print("=" * 80)
            print(f"🔍 Request received: {path}")
            print(f"📝 Original SQL: {sql}")
            
            # Return different optimization formats based on API endpoint
            if path == '/optimize_set':
                optimized_sql = self.optimize_with_set(sql)
            elif path == '/optimize_hint':
                optimized_sql = self.optimize_with_pg_hint_plan(sql)
            elif path == '/optimize_join_order':
                optimized_sql = self.optimize_join_order(sql)
            else:
                # Default: no optimization, return original SQL
                optimized_sql = sql
            
            # Print optimization result
            if optimized_sql != sql:
                print("✨ Optimized SQL:")
                print(optimized_sql)
            
            print("=" * 80)
            print()
            
            # Build response
            response = {
                "original_sql": sql,
                "optimized_sql": optimized_sql,
                "optimization_applied": optimized_sql != sql,
                "endpoint": path
            }
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.send_error(500, str(e))
    
    def optimize_with_set(self, sql):
        """
        Method 1: Use SET command format
        SET enable_mergejoin TO off; SET enable_indexscan TO off; + sql
        """
        hints = []
        
        # Add SET commands based on SQL features
        if re.search(r'\bJOIN\b', sql, re.IGNORECASE):
            hints.append("SET enable_mergejoin TO off")
            hints.append("SET enable_hashjoin TO off")
            hints.append("SET enable_nestloop TO on")
            print("📌 [SET Mode] Added JOIN optimization")
        
        if re.search(r'\bWHERE\b', sql, re.IGNORECASE):
            hints.append("SET enable_seqscan TO off")
            hints.append("SET enable_indexscan TO on")
            print("📌 [SET Mode] Added index scan")
        
        if re.search(r'\bORDER\s+BY\b', sql, re.IGNORECASE):
            hints.append("SET work_mem TO '128MB'")
            print("📌 [SET Mode] Increased work_mem")
        
        # Combine SET commands with SQL
        if hints:
            return "; ".join(hints) + "; " + sql
        return sql
    
    def optimize_with_pg_hint_plan(self, sql):
        """
        Method 2: Use pg_hint_plan comment format
        /*+ HashJoin(t mk mi) Leading(((t mk) mi)) */ + sql
        """
        hints = []
        
        # Extract table aliases
        tables = self.extract_table_aliases(sql)
        
        if len(tables) >= 2:
            # Multi-table JOIN - add join method and order hints
            if len(tables) == 2:
                hints.append(f"HashJoin({' '.join(tables)})")
                hints.append(f"Leading(({tables[0]} {tables[1]}))")
            elif len(tables) >= 3:
                # 3 tables or more
                hints.append(f"HashJoin({' '.join(tables)})")
                # Specify join order: ((t1 t2) t3)
                leading = f"((({tables[0]} {tables[1]}) {tables[2]}"
                for t in tables[3:]:
                    leading += f") {t}"
                leading += ")" * (len(tables) - 2)
                hints.append(f"Leading({leading})")
            print(f"📌 [HINT Mode] Added join strategy: {tables}")
        
        # Add scan method hints
        if re.search(r'\bWHERE\b', sql, re.IGNORECASE):
            for table in tables[:3]:  # Use BitmapScan for first 3 tables
                hints.append(f"BitmapScan({table})")
            print(f"📌 [HINT Mode] Added BitmapScan")
        
        # Combine hints
        if hints:
            hint_comment = "/*+ " + " ".join(hints) + " */ "
            return hint_comment + sql
        return sql
    
    def optimize_join_order(self, sql):
        """
        Method 3: Optimize join order specifically
        /*+ Leading(((t mk) mi)) */
        """
        tables = self.extract_table_aliases(sql)
        
        if len(tables) >= 2:
            # Construct join order
            if len(tables) == 2:
                leading = f"({tables[0]} {tables[1]})"
            else:
                # Left-deep tree: ((t1 t2) t3) t4)
                leading = f"(({tables[0]} {tables[1]})"
                for t in tables[2:]:
                    leading = f"({leading} {t})"
            
            hint = f"/*+ Leading({leading}) */"
            print(f"📌 [JOIN ORDER] Optimized order: {leading}")
            return hint + " " + sql
        
        return sql
    
    def extract_table_aliases(self, sql):
        """
        Extract table aliases from SQL
        Simple implementation: find table names/aliases after FROM and JOIN
        """
        tables = []
        
        # Match FROM table alias or JOIN table alias
        pattern = r'\b(?:FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?(\w+)'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        
        for match in matches:
            # match[0] is table name, match[1] is alias
            alias = match[1] if match[1] else match[0]
            if alias.lower() not in ['on', 'where', 'group', 'order', 'limit']:
                tables.append(alias)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tables = []
        for t in tables:
            if t not in seen:
                seen.add(t)
                unique_tables.append(t)
        
        return unique_tables
    
    def log_message(self, format, *args):
        # Suppress HTTP logging
        pass

if __name__ == '__main__':
    port = 8080
    server = HTTPServer(('0.0.0.0', port), MoLQOHandler)
    
    print("🚀 MoLQO Test Server Started")
    print(f"📡 Listening on port: {port}")
    print()
    print("📋 Available API endpoints:")
    print("   1. /optimize_set           - SET command format")
    print("   2. /optimize_hint          - pg_hint_plan comment format")
    print("   3. /optimize_join_order    - JOIN order optimization")
    print("=" * 80)
    print("Waiting for SQL queries...")
    print()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server shutdown")
        server.shutdown()
