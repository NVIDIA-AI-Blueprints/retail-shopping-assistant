# Shopping Assistant UI

A modern React TypeScript application for the NVIDIA Shopping Assistant demo.

## Overview

This UI provides a clean, responsive interface for interacting with the AI-powered shopping assistant. It features real-time chat, image upload capabilities, and a modern design built with React, TypeScript, and Tailwind CSS.

## Architecture

### Components Structure

```
src/
├── components/
│   ├── chatbox/
│   │   ├── ChatMessage.tsx      # Individual message display
│   │   ├── SafeHTML.tsx         # XSS-safe HTML rendering
│   │   ├── Loader.tsx           # Loading spinner
│   │   └── chatbox.tsx          # Main chat interface
│   ├── Apparel.tsx              # Product image display
│   ├── Navbar.tsx               # Navigation header
│   └── Footer.tsx               # Footer component
├── config/
│   └── config.ts                # Centralized configuration
├── types/
│   └── index.ts                 # TypeScript type definitions
├── utils/
│   └── index.ts                 # Utility functions
├── hooks/
│   └── useChat.ts               # Custom chat hook (WIP)
└── assets/                      # Static assets
```

### Key Features

- **Type Safety**: Full TypeScript implementation with proper type definitions
- **Configuration Management**: Centralized config for easy customization
- **Error Handling**: Comprehensive error handling with user feedback
- **Image Upload**: Secure image upload with validation
- **Real-time Streaming**: Live message streaming from the backend
- **Responsive Design**: Mobile-friendly interface
- **Accessibility**: ARIA labels and semantic HTML

## Development

### Prerequisites

- Node.js 16+
- npm or yarn

### Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start development server:
   ```bash
   npm start
   ```

3. Build for production:
   ```bash
   npm run build
   ```

### Available Scripts

- `npm start` - Start development server
- `npm run build` - Build for production
- `npm test` - Run tests
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint issues
- `npm run format` - Format code with Prettier

## Configuration

The application uses a centralized configuration system in `src/config/config.ts`. Key configuration options:

- **API Settings**: Backend URLs and endpoints
- **UI Settings**: Default images, categories, and styling
- **Feature Flags**: Enable/disable features like guardrails and image upload
- **File Upload**: Size limits and allowed file types

## Type Definitions

All TypeScript types are defined in `src/types/index.ts`:

- `MessageData` - Chat message structure
- `ApiRequest/ApiResponse` - API communication types
- `FileUploadResult` - Image upload handling
- `ErrorState` - Error handling types

## Utility Functions

Common utilities in `src/utils/index.ts`:

- File conversion (base64 ↔ blob)
- User session management
- API request helpers
- File validation
- Download utilities

## Styling

The application uses:
- **Tailwind CSS** for utility-first styling
- **Material-UI** for component library
- **Emotion** for styled components
- **Custom CSS** for chat-specific styling

## Security

- **XSS Protection**: DOMPurify for HTML sanitization
- **File Validation**: Strict file type and size validation
- **Input Sanitization**: All user inputs are properly sanitized

## Performance

- **Code Splitting**: Lazy loading of components
- **Memoization**: React.memo for expensive components
- **Efficient Rendering**: Optimized re-renders with proper keys
- **Streaming**: Real-time message streaming without blocking

## Testing

The application includes:
- Unit tests for utility functions
- Component testing with React Testing Library
- Integration tests for API communication
- E2E tests for critical user flows

## Deployment

The application can be deployed using:
- Docker containers
- Static hosting (Netlify, Vercel)
- Traditional web servers

## Contributing

1. Follow TypeScript best practices
2. Use proper error handling
3. Add tests for new features
4. Update documentation
5. Follow the existing code style

## Troubleshooting

### Common Issues

1. **TypeScript Errors**: Ensure all dependencies are properly typed
2. **Build Failures**: Check for missing dependencies
3. **API Connection**: Verify backend is running and accessible
4. **Image Upload**: Check file size and type restrictions

### Debug Mode

Enable debug logging by setting `NODE_ENV=development` in your environment.

## License

Apache 2.0 License - see LICENSE file for details. 