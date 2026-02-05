using UnityEngine;
using Unity.InferenceEngine;

public class ClassifyImage : MonoBehaviour
{
    [SerializeField] private Texture2D inputImage;
    [SerializeField] private ModelAsset modelAsset;
    [SerializeField] private TextAsset labelsField;
    [SerializeField] private string targetObject;
    private float[] results;
    

    private Worker m_Worker;
    private Tensor m_Input;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    // Changing to IEnumerator to ensure initialization is complete
    private string[] labels = new string[0];

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    // Changing to IEnumerator to ensure initialization is complete
    System.Collections.IEnumerator Start()
    {
        // Wait one frame to ensure all systems are initialized
        yield return null;

        if (modelAsset == null)
        {
            Debug.LogError("ModelAsset is not assigned!");
            yield break;
        }

        if (labelsField != null)
        {
            var rawLabels = labelsField.text.Split(new char[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
            labels = new string[rawLabels.Length];
            for (int i = 0; i < rawLabels.Length; i++)
            {
                string line = rawLabels[i].Trim();
                // Check for WordNet ID format (e.g., "n01440764 tench")
                int spaceIndex = line.IndexOf(' ');
                if (spaceIndex > 0 && spaceIndex < line.Length - 1 && line.StartsWith("n"))
                {
                    labels[i] = line.Substring(spaceIndex + 1);
                }
                else
                {
                    labels[i] = line;
                }
            }
        }

        try
        {
            Debug.Log("Loading model...");
            var model = ModelLoader.Load(modelAsset);
            m_Worker = new Worker(model, BackendType.GPUCompute);

            // Create input tensor from texture
            // The model expects (Batch, Channels, Height, Width) with Height/Width = 224
            m_Input = new Tensor<float>(new TensorShape(1, 3, 224, 224));
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error during initialization: {e.Message}\n{e.StackTrace}");
            yield break;
        }

        // Run once on start
        Classify();
    }

    // Public method to be called from UI Button
    public void Classify()
    {
        if (m_Worker == null || m_Input == null)
        {
            Debug.LogError("Worker or Input Tensor not initialized!");
            return;
        }

        if (inputImage == null)
        {
            Debug.LogError("InputImage is not assigned!");
            return;
        }

        try
        {
            Debug.Log($"Converting texture to tensor. Shape: {m_Input.shape}");
            // TextureConverter will automatically resample the texture to match the tensor shape
            TextureConverter.ToTensor(inputImage, m_Input as Tensor<float>);

            Debug.Log("Running inference...");
            m_Worker.Schedule(m_Input);

            // Get output
            var outputTensor = m_Worker.PeekOutput() as Tensor<float>;
            
            if (outputTensor == null)
            {
                Debug.LogError("Output tensor is null! Check if the model is loaded correctly and has a float output.");
                return;
            }

            // Readback from GPU to CPU to access data
            using (var cpuTensor = outputTensor.ReadbackAndClone())
            {
                Debug.Log($"Inference complete. Output shape: {cpuTensor.shape}");
                
                // Copy to results array for inspection in Inspector
                float[] rawResults = cpuTensor.DownloadToArray();

                if (rawResults == null || rawResults.Length == 0)
                {
                     Debug.LogError("No results downloaded from tensor!");
                     return;
                }

                // Apply Softmax to get probabilities
                results = Softmax(rawResults);

                // Find the index with the highest probability
                float maxProb = -1f;
                int maxIndex = -1;

                for (int i = 0; i < results.Length; i++)
                {
                    if (results[i] > maxProb)
                    {
                        maxProb = results[i];
                        maxIndex = i;
                    }
                }

                string className = (maxIndex >= 0 && maxIndex < labels.Length) ? labels[maxIndex] : "Unknown";
                
                Debug.Log($"<color=green>Predicted Class: {className} (Index: {maxIndex}) with Confidence: {maxProb:P2}</color>");

                // Check for target object
                if (!string.IsNullOrEmpty(targetObject))
                {
                    bool found = false;
                    for (int i = 0; i < labels.Length; i++)
                    {
                        // Split by comma to check synonyms (e.g. "goldfish, Carassius auratus")
                        string[] synonyms = labels[i].Split(',');
                        foreach (var synonym in synonyms)
                        {
                            if (synonym.Trim().Equals(targetObject.Trim(), System.StringComparison.OrdinalIgnoreCase))
                            {
                                float targetConfidence = results[i];
                                Debug.Log($"<color=cyan>Target '{labels[i]}' (Index: {i}) Confidence: {targetConfidence:P2}</color>");
                                found = true;
                                break;
                            }
                        }
                        if (found) break;
                    }
                    
                    if (!found)
                    {
                        Debug.LogWarning($"Target object '{targetObject}' not found in labels list.");
                    }
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error during inference: {e.Message}\n{e.StackTrace}");
        }
    }

    private float[] Softmax(float[] inputs)
    {
        float[] outputs = new float[inputs.Length];
        float max = float.MinValue;
        
        // Find max for numerical stability
        foreach (var val in inputs)
            if (val > max) max = val;

        float sum = 0f;
        for (int i = 0; i < inputs.Length; i++)
        {
            outputs[i] = Mathf.Exp(inputs[i] - max);
            sum += outputs[i];
        }

        for (int i = 0; i < outputs.Length; i++)
        {
            outputs[i] /= sum;
        }

        return outputs;
    }

    void OnDisable()
    {
        m_Worker?.Dispose();
        m_Input?.Dispose();
    }
}
