const express = require('express')
const multer = require('multer')
const jpeg = require('jpeg-js')
const dotenv= require('dotenv')
const tf = require('@tensorflow/tfjs-node')
const nsfw = require('nsfwjs')
const cors = require('cors')
const axios = require('axios')
const app = express()
const upload = multer()
const sharp = require('sharp');
app.use(cors());
app.use(express.json());
app.use(express.urlencoded());
let _model


const convert = async (img) => {
  // Decoded image in UInt8 Byte array
  const image = await jpeg.decode(img, true)

  const numChannels = 3
  const numPixels = image.width * image.height
  const values = new Int32Array(numPixels * numChannels)

  for (let i = 0; i < numPixels; i++)
    for (let c = 0; c < numChannels; ++c)
      values[i * numChannels + c] = image.data[i * 4 + c]

  return tf.tensor3d(values, [image.height, image.width, numChannels], 'int32')
}
app.get('/', (req, res) => {
  res.status(200).json("DevzCorner Welcome's You")
})
app.post('/nsfw', upload.single('image'), async (req, res) => {
  if (!req.file) res.status(400).send('Missing image multipart/form-data')
  else {
    const image = await convert(req.file.buffer)
    
    const predictions = await _model.classify(image)
    image.dispose()
    res.json(predictions)
  }
})
app.post('/testmodel',async(req, res) => {
  const { url } = req.body;
  
if(!url)
{

res.status(400).send('Missing image url');
}
const pic = await axios.get(url, {
  responseType: 'arraybuffer',
})
const model = await nsfw.load()
// const img = new Image();
// img.crossOrigin = "anonymous";
// img.src = url
// const image = await tf.node.decodeImage(pic.data,3)
  const predictions = await model.classify(url)
  image.dispose() // Tensor memory must be managed explicitly (it is not sufficient to let a tf.Tensor go out of scope for its memory to be released).
  console.log(predictions)
  res.json({predictions})
// nsfwjs.load().then((model) => {
//   // Classify the image.
//   model.classify(req.body.url).then((predictions) => {
//     res.json(pic)
//   });
// });
})
const load_model = async () => {
  _model = await nsfw.load()
}

// Keep the model in memory, make sure it's loaded only once
const PORT = process.env.PORT || 8000;
load_model().then(() => app.listen(PORT)).catch((error) => console.log(`${error} did not connect`));
