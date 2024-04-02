// UploadComponent.jsx
import { Box, Button } from '@chakra-ui/react';

const UploadComponent = ({ onUpload }) => {
  const handleUpload = (event) => {
    const files = event.target.files;
    onUpload(files);
  };

  return (
    <Box>
      <input type="file" onChange={handleUpload} />
    </Box>
  );
};

export default UploadComponent;
