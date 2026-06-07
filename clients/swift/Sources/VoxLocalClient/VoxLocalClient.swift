import Foundation

public struct TTSRequest: Codable {
    public let text: String
    public let language: String

    public init(text: String, language: String) {
        self.text = text
        self.language = language
    }
}

public struct AudioChunk: Codable {
    public let type: String
    public let sequence: Int
    public let sampleRate: Int
    public let channels: Int
    public let encoding: String
    public let `final`: Bool
    public let sourceChunk: Int?
    public let audioBase64: String

    enum CodingKeys: String, CodingKey {
        case type, sequence
        case sampleRate = "sample_rate"
        case channels, encoding
        case `final`
        case sourceChunk = "source_chunk"
        case audioBase64 = "audio_base64"
    }

    public func decodePCM16() -> Data? {
        guard let data = Data(base64Encoded: audioBase64) else { return nil }
        return data
    }
}

public struct TranscriptionResponse: Codable {
    public let text: String
}

public enum VoxLocalError: Error, LocalizedError {
    case requestFailed(statusCode: Int, message: String)
    case streamError(message: String)
    case invalidURL

    public var errorDescription: String? {
        switch self {
        case .requestFailed(let code, let msg):
            return "VoxLocal request failed (\(code)): \(msg)"
        case .streamError(let msg):
            return "VoxLocal stream error: \(msg)"
        case .invalidURL:
            return "Invalid VoxLocal server URL"
        }
    }
}

public class VoxLocalClient {
    private let baseURL: URL

    public init(baseUrl: String) throws {
        guard let url = URL(string: baseUrl) else {
            throw VoxLocalError.invalidURL
        }
        self.baseURL = url
    }

    public func synthesize(text: String, language: String) async throws -> Data {
        let url = baseURL.appendingPathComponent("v1/tts")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = TTSRequest(text: text, language: language)
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            let message = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw VoxLocalError.requestFailed(statusCode: statusCode, message: message)
        }
        return data
    }

    public func transcribe(audio: Data, language: String = "auto") async throws -> String {
        var urlComponents = URLComponents(url: baseURL.appendingPathComponent("v1/stt"), resolvingAgainstBaseURL: false)!
        urlComponents.queryItems = [URLQueryItem(name: "language", value: language)]

        var request = URLRequest(url: urlComponents.url!)
        request.httpMethod = "POST"
        request.setValue("audio/wav", forHTTPHeaderField: "Content-Type")
        request.httpBody = audio

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            let message = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw VoxLocalError.requestFailed(statusCode: statusCode, message: message)
        }

        let result = try JSONDecoder().decode(TranscriptionResponse.self, from: data)
        return result.text
    }

    public func stream(text: String, language: String) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let url = self.baseURL.appendingPathComponent("v1/tts/stream")
                    var request = URLRequest(url: url)
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")

                    let body = TTSRequest(text: text, language: language)
                    request.httpBody = try JSONEncoder().encode(body)

                    let (bytes, response) = try await URLSession.shared.bytes(for: request)

                    guard let httpResponse = response as? HTTPURLResponse,
                          httpResponse.statusCode == 200 else {
                        let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
                        throw VoxLocalError.requestFailed(statusCode: statusCode, message: "Stream failed")
                    }

                    var buffer = ""
                    for try await byte in bytes {
                        buffer.append(Character(UnicodeScalar(byte)))
                        while let newlineIndex = buffer.firstIndex(of: "\n") {
                            let line = String(buffer[buffer.startIndex...newlineIndex])
                            buffer = String(buffer[buffer.index(after: newlineIndex)...])

                            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                            guard !trimmed.isEmpty else { continue }

                            if let data = trimmed.data(using: .utf8) {
                                let chunk = try JSONDecoder().decode(AudioChunk.self, from: data)
                                continuation.yield(chunk)
                                if chunk.final {
                                    continuation.finish()
                                    return
                                }
                            }
                        }
                    }

                    if !buffer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                       let data = buffer.data(using: .utf8) {
                        let chunk = try JSONDecoder().decode(AudioChunk.self, from: data)
                        continuation.yield(chunk)
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
