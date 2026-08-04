/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    proxyClientMaxBodySize: '10gb',
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8000/api/:path*",
      },
      {
        source: "/outputs/:path*",
        destination: "http://127.0.0.1:8000/outputs/:path*",
      },
    ];
  },
};

export default nextConfig;