using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;


public class captureImg : MonoBehaviour
{
    // Start is called before the first frame update
    public int FileCounter;

    public Camera Cam;
    public Texture2D image;

    public string save_img_name;

    WaitForEndOfFrame frameEnd = new WaitForEndOfFrame();

    Rect rt;

    void Start(){
         Cam = GetComponent<Camera>();
         rt = new Rect(0, 0, Cam.pixelWidth, Cam.pixelHeight);
        //  Debug.Log(Cam.targetTexture.width);
        //  image = new Texture2D(Cam.target)
    }

    // Update is called once per frame
    void Update()
    {
       if(Input.GetKeyDown (KeyCode.Return))
	    {
		    Debug.Log("Enter");
            StartCoroutine(CamCapture(save_img_name));
	    }
    }

    public IEnumerator CamCapture(string imageName)
    {
        Camera cam = GetComponent<Camera>();
        yield return frameEnd;
        var path = "/img_dataset/" + imageName + ".png";
        // Debug.Log(path);
        var currentRT = RenderTexture.active;
        RenderTexture.active = cam.targetTexture;
        // Make a new texture and read the active Render Texture into it.
        image = new Texture2D(cam.pixelWidth, cam.pixelHeight);
        // cam.Render();
        
        image.ReadPixels(rt, 0, 0);
        image.Apply();
        // encode to PNG
        byte[] _bytes = image.EncodeToPNG();
        // save file
        System.IO.File.WriteAllBytes(path, _bytes);
        // set render texture back to default
        RenderTexture.active = currentRT;

        Destroy(image);
        Debug.Log("Saving File");
    }

}

