const defaultEslintConfig = [
  {
    files: ["**/*.{js,jsx,ts,tsx}"],
    ignores: ["node_modules"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },
      },
    },
    plugins: {},
    rules: {
      // Add any default rules you'd like to enforce
    },
  },
];