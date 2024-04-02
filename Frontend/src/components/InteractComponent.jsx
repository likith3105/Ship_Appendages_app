// InteractComponent.jsx
import { Box, Button } from '@chakra-ui/react';

const InteractComponent = ({ onSubmit }) => {
  const handleSubmit = () => {
    // Handle form submission
    onSubmit();
  };

  return (
    <Box>
      <Button onClick={handleSubmit}>Submit</Button>
    </Box>
  );
};

export default InteractComponent;
