require('dotenv').config();
const Hapi = require('@hapi/hapi');
const tf = require('@tensorflow/tfjs-node');
const { v4: uuidv4 } = require('uuid');
const { Firestore } = require('@google-cloud/firestore');

class InputError extends Error {
  constructor(message) {
    super(message);
    this.name = "InputError";
    this.statusCode = 400;
  }
}

// Load TensorFlow Model
const model_url = 'https://storage.googleapis.com/mlgc-bucket-ch2/model/model.json';
async function loadModel() {
  return tf.loadGraphModel(model_url);
}

// Preprocess Image Function
const preprocessImage = async (imageBuffer, targetHeight, targetWidth) => {
  const imageUint8Array = new Uint8Array(imageBuffer);
  const decodedImage = tf.node.decodeImage(imageUint8Array, 3);
  const resizedImage = tf.image.resizeBilinear(decodedImage, [targetHeight, targetWidth]);
  return resizedImage.expandDims(0);
};

async function predictClassification(model, imageBuffer) {
  try {
    const inputTensor = await preprocessImage(imageBuffer, 224, 224);
    const prediction = model.predict(inputTensor);
    const score = prediction.dataSync()[0];
    const label = score > 0.5 ? 'Cancer' : 'Non-cancer';
    const suggestion = label === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';
    return { label, suggestion };
  } catch (error) {
    console.error('Prediction error:', error);
    throw new InputError('Terjadi kesalahan dalam melakukan prediksi');
  }
}

async function storeData(id, data) {
  const db = new Firestore();
  const predictCollection = db.collection('predictions');
  return predictCollection.doc(id).set(data);
}

async function getAllData() {
  const db = new Firestore();
  const predictCollection = db.collection('predictions');
  const allData = await predictCollection.get();
  return allData.docs.map((doc) => ({
    id: doc.id,
    ...doc.data(),
  }));
}

(async () => {
  const server = Hapi.server({
    port: process.env.PORT || 8080,
    host: '0.0.0.0',
    routes: {
      cors: { origin: ['*'] },
    },
  });

  const model = await loadModel();
  server.app.model = model;

  server.route([
    // API: Predict
    {
      method: 'POST',
      path: '/predict',
      options: {
        payload: {
          output: 'stream',
          parse: true,
          multipart: true,
          maxBytes: 1000000, // Limit 1MB
        },
      },
      handler: async (request, h) => {
        try {
          const { model } = request.server.app;
          const file = request.payload.image;
  
          // Validasi: Periksa apakah file ada dan dapat dibaca
          if (!file || typeof file._readableState === 'undefined') {
            throw new InputError('No image file uploaded or file is invalid');
          }
  
          // Validasi: Pastikan ukuran file tidak melebihi batas maksimum
          if (file._data && file._data.length > 1000000) {
            throw new InputError('File size exceeds the 1MB limit');
          }
  
          const imageBuffer = await new Promise((resolve, reject) => {
            const chunks = [];
            file.on('data', (chunk) => chunks.push(chunk));
            file.on('end', () => resolve(Buffer.concat(chunks)));
            file.on('error', (err) => reject(err));
          });
  
          const { label, suggestion } = await predictClassification(model, imageBuffer);
  
          const id = uuidv4();
          const createdAt = new Date().toISOString();
          const data = { id, result: label, suggestion, createdAt };
  
          await storeData(id, data);
  
          return h.response({
            status: 'success',
            message: 'Model is predicted successfully',
            data,
          }).code(201);
        } catch (error) {
          if (error instanceof InputError) {
            return h.response({
              status: 'fail',
              message: error.message,
            }).code(error.statusCode);
          }
  
          console.error('Unexpected error:', error);
          return h.response({
            status: 'fail',
            message: 'An unexpected error occurred',
          }).code(500);
        }
      },
    },
  ]);
  
  server.ext('onPreResponse', (request, h) => {
    const response = request.response;

    if (response instanceof InputError) {
      return h.response({
        status: 'fail',
        message: response.message,
      }).code(response.statusCode);
    }

    if (response.isBoom) {
      return h.response({
        status: 'fail',
        message: response.output.payload.message,
      }).code(response.output.statusCode);
    }

    return h.continue;
  });

  await server.start();
  console.log(`Server is running on ${server.info.uri}`);
})();
