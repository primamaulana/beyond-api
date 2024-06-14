const predictClassification = require('../services/inferenceService');
const crypto = require('crypto');
const { storeData, predictionsCollection } = require('../services/storeData')
 
async function postPredictHandler(request, h) {
  const { image } = request.payload;
  const { model } = request.server.app;
 
  const { confidenceScore, label } = await predictClassification(model, image);
  const id = crypto.randomUUID();
  const createdAt = new Date().toISOString();
 
  const data = {
    "id": id,
    "result": label,
    "score": confidenceScore,
    "createdAt": createdAt
  }

  await storeData(id, data);
  const response = h.response({
    status: 'success',
    message: 'Model is predicted successfully',
    data
  })
  response.code(201);
  return response;
}

async function getPredictHistories(request, h) {
  const histories = (await predictionsCollection.get()).docs.map((doc) =>
    doc.data()
  );
  const data = histories.map((item) => ({
    id: item.id,
    history: item,
  }));
  return h.response({ status: 'success', data }).code(200);
}
 
module.exports = { postPredictHandler, getPredictHistories };