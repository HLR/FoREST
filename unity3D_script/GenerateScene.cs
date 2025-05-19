using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using static System.Math;

public class GenerateScene : MonoBehaviour
{   

    [System.Serializable]
    public class SceneInfoData{
            public string imgID;
            public string bgName;
            public gameObjectInfo[] obj = new gameObjectInfo[2];
            public void setObjectInfo(int objIndex, GameObject curObj, string objName)
            {
                // Debug.Log(curObj.GetComponent<GetBoundingBox>().get_bound(512, 512));
                obj[objIndex] = new gameObjectInfo();
                obj[objIndex].SetBoundingBox(curObj.GetComponent<GetBoundingBox>().get_bound(512, 512));
                obj[objIndex].degRotation = curObj.transform.localEulerAngles.y;
                obj[objIndex].name = objName;
            }
        }
    [System.Serializable]
    public class SceneInfo{
        public string id;
        public string context;
        public string[] label;
        public string obj1;
        public string obj2;

        public string obj2_dir;
        public string relation;

    }

    [System.Serializable]
    public class SceneObj{
            public string name;
            public GameObject[] objList;
        }

    [System.Serializable]
    public class Dataset{
            public string dataset;
            public SceneInfo[] data;

            public static Dataset CreateFromJSON(string jsonString)
            {
            return JsonUtility.FromJson<Dataset>(jsonString);
            }
        }

    [System.Serializable]
    public class gameObjectInfo{
            public int xMin;
            public int xMax;
            public int yMin;
            public int yMax;
            public float degRotation;
            public string name;

            public void SetBoundingBox(int[] boundingBox){
                if(boundingBox.Length < 4){
                    return;
                }
                xMin = Max(0, boundingBox[0]);
                yMin = Max(0, boundingBox[1]);
                xMax = Min(512, boundingBox[2]);
                yMax = Min(512, boundingBox[3]);
            }
        }


    [System.Serializable]
    public class SceneInfoJson{
        public SceneInfoData[] info;
    }


    public GameObject mainCam;

    public TextAsset dataSet;

    public bool captureImg;

    public bool generateCollision;

    public Vector3 lowestLocation;

    public Vector3 highestLocation;

    public bool generateOutOfScreen;

    public int runGenerate;

    public GameObject[] allSpawnObj;

    public GameObject[] backgrounds;

    public int randomSeed;

    public SceneObj[] allObject;

    public Dataset dataObj;

    public List<SceneInfoData> allGeneratedData;

    public int repeatedImage;

    WaitForSeconds shortWait = new WaitForSeconds(0.0000000000001f);

    SceneInfo[] objData;

    public string saveJsonFile;

    private string covertJson;

    public SceneInfoJson jsonData; 

    public Vector2[] degree_range;


    // Start is called before the first frame update
    void Start()
    {
        dataObj = Dataset.CreateFromJSON(dataSet.text);
        allGeneratedData = new List<SceneInfoData>();
        objData = dataObj.data;
        generateOutOfScreen = false;
        generateCollision = false;
        jsonData = new SceneInfoJson();
       
        
    }

    
    // Update is called once per frame
    void Update()
    {
        if(captureImg){
            captureImg = false;
            StartCoroutine(CaptureImage(mainCam));
            // StartCoroutine(Test());
        }
        
    }

    private void MoveObject(GameObject obj, Vector3 movement){
        obj.transform.Translate(movement);
    }

    IEnumerator CaptureImage(GameObject camera)
    {
        Random.InitState(randomSeed);
        Debug.Log("Start Generate!!");
        allSpawnObj = new GameObject[4];
        for(; runGenerate < objData.Length; runGenerate++){
            yield return StartCoroutine(GenerateObjects(allSpawnObj, objData[runGenerate]));
        }
        Debug.Log("Finish");
        jsonData.info = allGeneratedData.ToArray();
        covertJson = JsonUtility.ToJson(jsonData, true);
        File.WriteAllText(saveJsonFile, covertJson);
    }

    IEnumerator GenerateObjects(GameObject[] objectList, SceneInfo curScene){
        int count = repeatedImage;
        int degree_index = count;
        while(count > 0){
            count -= 1;
            // Disable any background
            for(int idx = 0; idx < backgrounds.Length; idx++){
                backgrounds[idx].SetActive(false);
            }

            // Create object 1
            while(true){
                foreach(SceneObj obj in allObject){
                    if(obj.name == curScene.obj2){
                        GameObject generatedObj = obj.objList[Random.Range(0, obj.objList.Length)];
                        objectList[0] = Instantiate(generatedObj, new Vector3(Random.Range(lowestLocation[0], highestLocation[0]), 0,  Random.Range(lowestLocation[2], highestLocation[2])), generatedObj.transform.rotation);
                        
                        switch(curScene.obj2_dir){
                            case "front":
                            degree_index = 0; break;
                            case "left":
                            degree_index = 1; break;
                            case "back":
                            degree_index = 2; break;
                            case "right":
                            degree_index = 3; break;
                            case "nan":
                            degree_index = count; break;
                        }

                        objectList[0].transform.Rotate(0, Random.Range(degree_range[degree_index].x, degree_range[degree_index].y), 0, Space.World);
                    }
                }


            // Create object 2
                foreach(SceneObj obj in allObject){
                    if(obj.name == curScene.obj1){
                        GameObject generatedObj = obj.objList[Random.Range(0, obj.objList.Length)];
                        objectList[1] = Instantiate(generatedObj, objectList[0].transform.position, objectList[0].transform.rotation);

                        // Debug.Log("End with: " + generateCollision);
                    }
                }
                // Translate object based on object 1 position
                float movement = Random.Range(2.0f, 20.0f);
                
                var spaceRef = Space.World;
                if(curScene.label[0] == "external intrinsic"){
                    if(curScene.relation != "front" && curScene.relation != "back"){
                        movement = -movement;
                    }
                    spaceRef = Space.Self;
                }
                switch(curScene.relation){
                    case "left":
                    objectList[1].transform.Translate(new Vector3(movement, 0, 0), spaceRef);
                    objectList[1].transform.position = new Vector3(objectList[1].transform.position.x, objectList[1].transform.position.y, objectList[1].transform.position.z + Random.Range(-2.0f, 2.0f));
                    break;

                    case "right":
                    objectList[1].transform.Translate(new Vector3(-movement, 0, 0), spaceRef);
                    objectList[1].transform.position = new Vector3(objectList[1].transform.position.x, objectList[1].transform.position.y, objectList[1].transform.position.z + Random.Range(-2.0f, 2.0f));
                    break;

                    case "above":
                    objectList[1].transform.Translate(new Vector3(0, movement, 0), spaceRef);
                    objectList[1].transform.position = new Vector3(objectList[1].transform.position.x + Random.Range(-2.0f, 2.0f), objectList[1].transform.position.y, objectList[1].transform.position.z + Random.Range(-2.0f, 2.0f));
                    break;

                    case "below":
                    objectList[1].transform.Translate(new Vector3(0, -movement, 0), spaceRef);
                    objectList[1].transform.position = new Vector3(objectList[1].transform.position.x + Random.Range(-2.0f, 2.0f), objectList[1].transform.position.y, objectList[1].transform.position.z + Random.Range(-2.0f, 2.0f));
                    break;

                    case "front":
                    objectList[1].transform.Translate(new Vector3(0, 0, movement), spaceRef);
                    objectList[1].transform.position = new Vector3(objectList[1].transform.position.x + Random.Range(-2.0f, 2.0f), objectList[1].transform.position.y, objectList[1].transform.position.z);
                    break;

                    case "back":
                    objectList[1].transform.Translate(new Vector3(0, 0, -movement), spaceRef);
                    objectList[1].transform.position = new Vector3(objectList[1].transform.position.x + Random.Range(-2.0f, 2.0f), objectList[1].transform.position.y, objectList[1].transform.position.z);
                    break;
                }
                // Define the orientation of the locatum based on the direction
                if(curScene.obj1 == "cabinet"){
                    objectList[1].transform.rotation = Quaternion.identity;
                    objectList[1].transform.Rotate(0, Random.Range(-40.0f, 90.0f), 0, Space.World);
                }
                else{
                    objectList[1].transform.Rotate(0, Random.Range(0.0f, 360.0f), 0, Space.World);
                }
                
                yield return shortWait;

                objectList[0].GetComponent<OutOfCamCheck>().CameraCheck();
                objectList[1].GetComponent<OutOfCamCheck>().CameraCheck();

                yield return shortWait;

                // bool check_near_cam = (objectList[1].transform.position.z > highestLocation[2] || objectList[0].transform.position.z > highestLocation[2] || objectList[1].transform.position.x > highestLocation[0] || objectList[0].transform.position.x > highestLocation[0]);
                // bool check_far_cam = (objectList[1].transform.position.z < lowestLocation[2] || objectList[0].transform.position.z < lowestLocation[2] || objectList[1].transform.position.x < lowestLocation[0] || objectList[0].transform.position.x < lowestLocation[0]);
                if((generateCollision || generateOutOfScreen)){
                    Destroy(objectList[0]);
                    Destroy(objectList[1]);
                    generateCollision = false;
                    generateOutOfScreen = false;
                    // yield return shortWait;
                    continue;
                }
                // Debug.Log("Out of screen2:" + generateOutOfScreen);
                break;
                // yield return new WaitForSeconds(waitTime);
            }

            // Selected background            
            int randomBg = Random.Range(0, backgrounds.Length);
            backgrounds[randomBg].SetActive(true);
            yield return StartCoroutine(mainCam.GetComponent<captureImg>().CamCapture(curScene.id + "_img_" + count.ToString()));
            // Destroy all created objects
            // Getting all info before deleting object from scene

            var curSceneJson = new SceneInfoData();
            curSceneJson.imgID = curScene.id + "_img_" + count.ToString();
            curSceneJson.bgName = backgrounds[randomBg].name;
            curSceneJson.setObjectInfo(0, objectList[1], curScene.obj1);
            curSceneJson.setObjectInfo(1, objectList[0], curScene.obj2);

            allGeneratedData.Add(curSceneJson);

            for(int j=0;j<objectList.Length;j++){
                
                Destroy(objectList[j]);
            }
        }
        yield break; 
    }
}

