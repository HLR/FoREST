using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OutOfCamCheck : MonoBehaviour
{
    // Start is called before the first frame update
    Camera mainCamera;

    Collider objCollider;

    public GameObject centerPosition;

    public bool changeCenter;

    Renderer m_Renderer;

    int render_x = 512;
    int render_y = 512;

    public int out_of_frame_threshold = 5; // Can be out of frame by 5 pixel

    public int[] bounding_box;

    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        CameraCheck();
    }

    public void CameraCheck(){
   
        mainCamera = Camera.main;
        GameObject centerObject = gameObject;
            // Debug.Log(centerObject);
            if(changeCenter){
                centerObject = centerPosition;
            }
        objCollider = centerObject.GetComponent<Collider>();
        if(CheckOutOfCamera(mainCamera)){
            // Bit shift the index of the layer (8) to get a bit mask
            int layerMask = 1 << 8;

            // This would cast rays only against colliders in layer 8.
            // But instead we want to collide against everything except layer 8. The ~ operator does this, it inverts a bitmask.
            layerMask = ~layerMask;

            RaycastHit hit;
            
            // Does the ray intersect any objects
            if (Physics.Raycast(mainCamera.transform.position, centerObject.transform.position - mainCamera.transform.position, out hit, Mathf.Infinity, layerMask))
            {
                GameObject hitObject = hit.collider.gameObject;
                // Center of obj can be seen from camera
                if (hitObject == gameObject){
                    Debug.DrawRay(mainCamera.transform.position, centerObject.transform.position - mainCamera.transform.position, Color.red);

                    // Check bounding box within the frame
                    
                    bounding_box = GetComponent<GetBoundingBox>().get_bound(render_x, render_y);
                    // Debug.Log(this.name + ": " + bounding_box[0] + "," + bounding_box[1] + "," + bounding_box[2] + "," + bounding_box[3]);
                    if(bounding_box[0] >= -out_of_frame_threshold && bounding_box[1] >= -out_of_frame_threshold && bounding_box[2] <= render_x + out_of_frame_threshold && bounding_box[3] <= render_y + out_of_frame_threshold){
                        return; // Satisfy in-camera. So don't change the outOfScreen value
                    }
                    
                }
            }
        }

        // Does not satisfied in-camera condition
        GameObject.Find("GenerateScene").GetComponent<GenerateScene>().generateOutOfScreen = true; 
        
    }
    

    private bool CheckOutOfCamera(Camera cam){
        Plane[] cameraFrustum = GeometryUtility.CalculateFrustumPlanes(cam);
        
        var bounds = objCollider.bounds;
        bool insideCamera = GeometryUtility.TestPlanesAABB(cameraFrustum, bounds);
        // Debug.Log(gameObject + " inside Camera: " + insideCamera);
        return insideCamera;
    }
}
