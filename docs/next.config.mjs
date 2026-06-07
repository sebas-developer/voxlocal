import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  basePath: '/voxlocal',
  images: {
    unoptimized: true,
  },
};

export default withMDX(nextConfig);
