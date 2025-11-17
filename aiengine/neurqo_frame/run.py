# Standard library imports
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# Local/project imports
from common import get_config
from moqoe import MoQOEController


class MoQOEHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for MoQOE inference API."""

    moqoe_instance = None  # Class variable, set when server starts

    @classmethod
    def set_moqoe(cls, moqoe: MoQOEController):
        """Set MoQOE instance."""
        cls.moqoe_instance = moqoe

    @property
    def moqoe(self):
        """Get MoQOE instance."""
        return self.__class__.moqoe_instance

    def do_POST(self):
        # Parse request path
        path = urlparse(self.path).path

        # Read request body
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        request_json = json.loads(post_data.decode("utf-8"))

        # Get SQL query
        sql = request_json.get("sql", "").strip()

        try:
            print("=" * 80)
            print(f"Request received: {path}")
            print(f"Original SQL: {sql}")

            if not self.moqoe:
                raise ValueError("MoQOE instance not initialized")

            # Execute inference optimization
            optimized_sql, expert_name = self.moqoe.inference(sql)

            # Print optimization result
            if optimized_sql != sql:
                print("Optimized SQL:")
                print(optimized_sql)
            else:
                print("SQL not optimized")

            print("=" * 80)
            print()

            # Build response
            response = {
                "original_sql": sql,
                "optimized_sql": optimized_sql,
                "optimization_applied": optimized_sql != sql,
                "expert_name": expert_name if expert_name else "none",
                "endpoint": path,
            }

            # Send response
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            error_response = {
                "error": str(e),
                "original_sql": sql if "sql" in locals() else "",
                "optimized_sql": sql if "sql" in locals() else "",
                "optimization_applied": False,
                "expert_name": "cost-based optimizer",  # Fallback to cost-based optimizer on error
            }
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode("utf-8"))


if __name__ == "__main__":
    # Configuration - use get_config to get BaseConfig
    DATASET = "imdb"  # Default dataset, can be changed
    config = get_config(DATASET)

    # Initialize MoQOE instance
    buffer_path = "./models/buffer_imdb_ori.db"  # Default buffer path
    database_name = "imdb_ori"  # Default database name
    moqoe = MoQOEController(
        buffer_path=buffer_path,
        config=config,
        DATASET=DATASET,
        database_name=database_name,
    )

    # Set MoQOE instance to Handler
    MoQOEHandler.set_moqoe(moqoe)

    # Create HTTP server
    port = 8666
    server = HTTPServer(("0.0.0.0", port), MoQOEHandler)

    print("MoQOE Inference Server Started")
    print(f"Listening on port: {port}")
    print()
    print("Available API endpoint:")
    print("   POST /optimize     - SQL query optimization")
    print()
    print("Request format (JSON):")
    print('   {"sql": "SELECT * FROM table"}')
    print("=" * 80)
    print("Waiting for SQL queries...")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutdown")
        server.shutdown()
