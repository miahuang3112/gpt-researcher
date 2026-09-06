/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async rewrites() {
    const academicApi = process.env.ACADEMIC_API_URL || "http://127.0.0.1:8001";
    return [{
      source: "/api/academic/:path*",
      destination: `${academicApi}/api/academic/:path*`,
    }];
  },
};

export default nextConfig;
