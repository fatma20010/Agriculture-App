from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import uvicorn
import logging
from typing import Dict, Any
import io
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target API server
TARGET_API = "http://127.0.0.1:5002"

app = FastAPI()

# Add CORS middleware with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Add OPTIONS route handler for preflight requests
@app.options("/{path:path}")
async def options_handler(request: Request, path: str):
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH"
    response.headers["Access-Control-Allow-Headers"] = "*, Accept, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization"
    response.headers["Access-Control-Expose-Headers"] = "*"
    return response

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "PATCH"])
async def proxy_endpoint(request: Request, path: str):
    """
    Proxy all requests to the target API server
    """
    target_url = f"{TARGET_API}/{path}"
    logger.info(f"Proxying request: {request.method} {target_url}")

    # Get request body
    body = await request.body()
    
    # Get request headers
    headers = dict(request.headers)
    headers.pop("host", None)
    
    try:
        async with httpx.AsyncClient() as client:
            # Forward the request
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                params=request.query_params,
                cookies=request.cookies,
                follow_redirects=True,
                timeout=30.0
            )
            
            # Create response with same status code and headers
            content = response.content
            response_headers = dict(response.headers)
            
            # Ensure CORS headers are present in the response
            response_headers["Access-Control-Allow-Origin"] = "*"
            response_headers["Access-Control-Allow-Credentials"] = "true"
            
            return Response(
                content=content,
                status_code=response.status_code,
                headers=response_headers,
                media_type=response.headers.get("content-type")
            )
    except httpx.RequestError as exc:
        logger.error(f"Error proxying request: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail=f"Error proxying request: {str(exc)}")
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(exc)}")

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{TARGET_API}/", timeout=5.0)
            if response.status_code == 200:
                return {"status": "ok", "backend": "reachable"}
            else:
                return {"status": "warning", "backend": f"returned status {response.status_code}"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "backend": "unreachable", "error": str(e)}

if __name__ == "__main__":
    logger.info(f"Starting FastAPI proxy server for {TARGET_API}")
    uvicorn.run(app, host="0.0.0.0", port=8889) 