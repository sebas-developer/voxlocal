// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "VoxLocalClient",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(
            name: "VoxLocalClient",
            targets: ["VoxLocalClient"]
        ),
    ],
    targets: [
        .target(
            name: "VoxLocalClient"
        ),
    ]
)
