# Deepfake Video Detector API Documentation

This document provides comprehensive documentation for the Deepfake Video Detector REST API, including endpoints, request/response formats, authentication, and usage examples. From Hasif's Workspace.

## Base URL

```
http://localhost:8000
```

For production deployments, replace `localhost:8000` with your actual server address.

## API Version

Current API version: `v1`

All API endpoints are prefixed with `/api/v1/`

## Authentication

Currently, the API does not require authentication. For production deployments, consider implementing:
- API key authentication
- JWT tokens
- OAuth 2.0

## Content Types

- **Request Content-Type**: `multipart/form-data` for file uploads
- **Response Content-Type**: `application/json`

## Rate Limiting

- **Default Limit**: 60 requests per minute per IP address
- **Burst Limit**: 10 requests in quick succession
- **Headers**: Rate limit information is included in response headers

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages in JSON format.

### Error Response Format

```json
{
  "detail": "Error description",
  "error_code": "ERROR_CODE",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common HTTP Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `413 Payload Too Large`: File size exceeds limit
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## Endpoints

### 1. Root Endpoint

Get basic API information.

**Endpoint**: `GET /`

**Response**:
```json
{
  "message": "Deepfake Video Detector API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/api/v1/health"
}
```

### 2. Health Check

Check API health and status.

**Endpoint**: `GET /api/v1/health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "uptime": 3600.5
}
```

### 3. Analyze Video

Analyze a video file for deepfake detection.

**Endpoint**: `POST /api/v1/analyze-video`

**Request Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `video_file` | File | Yes | - | Video file to analyze (MP4, AVI, MOV, MKV, WMV) |
| `num_frames` | Integer | No | 5 | Number of frames to extract and analyze (1-50) |
| `enable_gradcam` | Boolean | No | true | Whether to generate Grad-CAM visualizations |
| `confidence_threshold` | Float | No | 0.5 | Confidence threshold for classification (0.0-1.0) |

**Request Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/analyze-video" \
  -F "video_file=@sample_video.mp4" \
  -F "num_frames=10" \
  -F "enable_gradcam=true" \
  -F "confidence_threshold=0.6"
```

**Response**:
```json
{
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "overall_prediction": "Real",
  "confidence_score": 0.87,
  "frame_predictions": [
    {
      "frame_number": 1,
      "prediction": "Real",
      "confidence": 0.89,
      "gradcam_available": true,
      "gradcam_path": "/api/v1/gradcam/550e8400-e29b-41d4-a716-446655440000/1"
    },
    {
      "frame_number": 2,
      "prediction": "Real",
      "confidence": 0.85,
      "gradcam_available": true,
      "gradcam_path": "/api/v1/gradcam/550e8400-e29b-41d4-a716-446655440000/2"
    }
  ],
  "processing_time": 2.34,
  "model_version": "efficientnet_b0_v1.0",
  "metadata": {
    "original_filename": "sample_video.mp4",
    "file_size": 15728640,
    "frames_extracted": 10,
    "video_duration": 5.2,
    "model_architecture": "EfficientNet-B0",
    "confidence_threshold": 0.6
  }
}
```

### 4. Get Grad-CAM Visualization

Retrieve Grad-CAM visualization for a specific frame.

**Endpoint**: `GET /api/v1/gradcam/{video_id}/{frame_number}`

**Path Parameters**:
- `video_id`: Unique video identifier from analysis response
- `frame_number`: Frame number (1-based indexing)

**Response**: PNG image file

**Example**:
```bash
curl "http://localhost:8000/api/v1/gradcam/550e8400-e29b-41d4-a716-446655440000/1" \
  --output gradcam_frame_1.png
```

### 5. Model Information

Get information about the loaded model.

**Endpoint**: `GET /api/v1/models/info`

**Response**:
```json
{
  "model_name": "DeepfakeDetector",
  "architecture": "EfficientNet-B0",
  "version": "efficientnet_b0_v1.0",
  "loaded": true,
  "input_size": [224, 224],
  "classes": ["Real", "Deepfake"]
}
```

## File Upload Specifications

### Supported Video Formats

- **MP4** (`.mp4`) - Recommended
- **AVI** (`.avi`)
- **MOV** (`.mov`)
- **MKV** (`.mkv`)
- **WMV** (`.wmv`)

### File Size Limits

- **Maximum file size**: 500 MB
- **Recommended size**: Under 100 MB for faster processing

### Video Specifications

- **Resolution**: Any resolution (will be resized to 224x224 for processing)
- **Duration**: Any duration (frames will be sampled)
- **Frame rate**: Any frame rate
- **Codecs**: Most common video codecs supported

## Usage Examples

### Python Example

```python
import requests

# Analyze video
url = "http://localhost:8000/api/v1/analyze-video"
files = {"video_file": open("sample_video.mp4", "rb")}
data = {
    "num_frames": 5,
    "enable_gradcam": True,
    "confidence_threshold": 0.5
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Prediction: {result['overall_prediction']}")
print(f"Confidence: {result['confidence_score']:.2%}")

# Download Grad-CAM visualization
if result['frame_predictions'][0]['gradcam_available']:
    gradcam_url = f"http://localhost:8000{result['frame_predictions'][0]['gradcam_path']}"
    gradcam_response = requests.get(gradcam_url)
    
    with open("gradcam_frame_1.png", "wb") as f:
        f.write(gradcam_response.content)
```

### JavaScript Example

```javascript
// Analyze video using FormData
const formData = new FormData();
formData.append('video_file', videoFile);
formData.append('num_frames', '5');
formData.append('enable_gradcam', 'true');
formData.append('confidence_threshold', '0.5');

fetch('http://localhost:8000/api/v1/analyze-video', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Prediction:', data.overall_prediction);
    console.log('Confidence:', data.confidence_score);
    
    // Display Grad-CAM images
    data.frame_predictions.forEach(frame => {
        if (frame.gradcam_available) {
            const img = document.createElement('img');
            img.src = `http://localhost:8000${frame.gradcam_path}`;
            document.body.appendChild(img);
        }
    });
})
.catch(error => console.error('Error:', error));
```

### cURL Examples

```bash
# Basic video analysis
curl -X POST "http://localhost:8000/api/v1/analyze-video" \
  -F "video_file=@video.mp4" \
  -F "num_frames=5"

# Advanced analysis with custom parameters
curl -X POST "http://localhost:8000/api/v1/analyze-video" \
  -F "video_file=@video.mp4" \
  -F "num_frames=10" \
  -F "enable_gradcam=true" \
  -F "confidence_threshold=0.7"

# Health check
curl "http://localhost:8000/api/v1/health"

# Model information
curl "http://localhost:8000/api/v1/models/info"
```

## Response Time Expectations

- **Health check**: < 100ms
- **Model info**: < 200ms
- **Video analysis**: 
  - 5 frames: 2-5 seconds
  - 10 frames: 4-8 seconds
  - 20 frames: 8-15 seconds

*Note: Processing times depend on video resolution, server hardware, and whether GPU acceleration is available.*

## Best Practices

### For Optimal Performance

1. **Video Quality**: Use high-quality videos for better accuracy
2. **File Size**: Keep videos under 100MB when possible
3. **Frame Count**: Use 5-10 frames for balance between speed and accuracy
4. **Batch Processing**: For multiple videos, process them sequentially

### For Production Use

1. **Error Handling**: Always check response status codes
2. **Timeout**: Set appropriate timeouts (30-60 seconds for video analysis)
3. **Rate Limiting**: Respect rate limits and implement backoff strategies
4. **Caching**: Cache results when appropriate
5. **Security**: Implement authentication and input validation

## Troubleshooting

### Common Issues

1. **File Too Large**: Reduce video file size or duration
2. **Unsupported Format**: Convert video to supported format
3. **Processing Timeout**: Reduce number of frames or video duration
4. **Model Not Loaded**: Check server logs and model availability

### Debug Information

Enable debug mode by setting `DEBUG=true` in environment variables for detailed error messages.

## OpenAPI Documentation

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These interfaces provide:
- Interactive API testing
- Detailed schema information
- Request/response examples
- Authentication testing (when implemented)

## Changelog

### Version 1.0.0
- Initial API release
- Video analysis endpoint
- Grad-CAM visualization support
- Health check and model info endpoints
- Comprehensive error handling
