using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CollisionCheck : MonoBehaviour
{
    public static bool collision;

    void Start()
    {
        collision = false;
    }
    // Start is called before the first frame update
    void OnCollisionEnter(Collision other)
    {
        GameObject.Find("GenerateScene").GetComponent<GenerateScene>().generateCollision = true;
        // Debug.Log("Trigger");
    }

    void OnCollisionExit(Collision other)
    {
        // collision = false;
        // GameObject.Find("GenerateScene").GetComponent<GenerateScene>().generateCollision = false;
    }

}
