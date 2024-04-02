import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Button, Input, Box, Card, CardBody, Divider, Heading, Stack, Image, Text } from "@chakra-ui/react";

import './App.css';

function App() {
  
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [uploadError, setUploadError] = useState(false); // State to track uploading status
  const [cnnData, setCnnData] = useState(null);
  const [vggnetData, setVggnetData] = useState(null);
  const [resnetData, setResnetData] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [description, setDescription] = useState("");

  const [data, setData] = useState([{}])

  useEffect(() =>{
      fetch("/train/cnn").then(
        res => res.json()
      ).then(
        data => {
            setData(data)
            console.log(data)
        }

      )
  },[])
  const fetchDataFromBackend = async (modelName) => {
    try {
      const response = await axios.get(`http://127.0.0.1:5000`);
      switch (modelName) {
        case 'cnn':
          setCnnData(response.data);
          break;
        case 'vggnet':
          setVggnetData(response.data);
          break;
        case 'resnet':
          setResnetData(response.data);
          break;
        default:
          break;
      }
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const handleFetchData = (modelName) => {
    fetchDataFromBackend(modelName);
  };





  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setUploadedImage(URL.createObjectURL(event.target.files[0])); // Set uploaded image for preview
  };

  const handleUpload = () => {
    setUploading(true);
    // Simulate upload process
    setTimeout(() => {
      setUploadSuccess(true);
      setTimeout(() => {
        setDescription("Crazing in metal surfaces refers to the formation of small cracks or fissures on the surface of a metal material. These cracks are typically very fine and may appear in a network-like pattern. Crazing can occur due to various factors such as stress, strain, thermal expansion or contraction, chemical exposure, or a combination of these factors.");
      }, 1200000); // 20 minutes delay
    }, 1000);
  };

  return (
    <Box height="100vh" display="flex" justifyContent="center" alignItems="center">
      <Card maxW='sm' p={5}>
        <CardBody>
          <Stack mt='6' spacing='3'>
            <Heading textAlign="center" size='md'>Model Evaluation</Heading>
          </Stack>
        </CardBody>
        <Divider />
        <Box p={4}>
          <Input type="file" onChange={handleFileChange} />
          <Button onClick={handleUpload} colorScheme="green" mt={4}>Upload</Button>
        </Box>
        <Divider />
        {uploadedImage && (
          <Box p={4}>
            <Image src={uploadedImage} alt="Uploaded" />
            <Text mt={2}>{description}</Text>
          </Box>
        )}
        <Divider />
        <Box p={4} display="flex" justifyContent="space-around">
          <Button colorScheme="blue" onClick={() => handleFetchData('cnn')}> CNN </Button>
          <Button colorScheme="blue" onClick={() => handleFetchData('vggnet')}> VGGNet  </Button>
          <Button colorScheme="blue" onClick={() => handleFetchData('resnet')}> ResNet </Button>
        </Box>
      </Card>
    </Box>
  );
}

export default App;
