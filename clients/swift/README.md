# VoxLocalClient

Swift client for the VoxLocal HTTP streaming API.

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/sebas-developer/voxlocal.git", from: "0.1.0")
]
```

Or in Xcode: File → Add Package Dependencies → enter the repository URL.

## Usage

```swift
import VoxLocalClient

let client = try VoxLocalClient(baseUrl: "http://127.0.0.1:8765")

// Text-to-speech
let audioData = try await client.synthesize(text: "Hola mundo", language: "es")

// Streaming TTS
for try await chunk in client.stream(text: "Hola mundo", language: "es") {
    let pcmData = chunk.decodePCM16()
    // Play audio chunk...
}

// Speech-to-text
let text = try await client.transcribe(audio: wavData, language: "auto")
```

## License

MIT
