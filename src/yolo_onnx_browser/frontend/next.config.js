/** @type {import('next').NextConfig} */
const NodePolyfillPlugin = require("node-polyfill-webpack-plugin");
const CopyPlugin = require("copy-webpack-plugin");

const nextConfig = {
  reactStrictMode: true,
  webpack: (config, options) => {
    config.resolve.extensions.push(".ts", ".tsx");

    config.resolve = {
      ...config.resolve,
      fallback: {
        fs: false,
        path: false,
        crypto: false,
      },
    };

    config.plugins.push(
      new NodePolyfillPlugin(),
      new CopyPlugin({
        patterns: [
          {
            from: "./node_modules/onnxruntime-web/dist/*.wasm",
            to: "static/chunks/app/[name][ext]",
          },
          {
            from: "./node_modules/onnxruntime-web/dist/*.wasm",
            to: "static/chunks/[name][ext]",
          },
        ],
      })
    );

    return config;
  },
};

module.exports = nextConfig;
