using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class GetBoundingBox : MonoBehaviour
{
    Collider m_Collider;
    Vector3 m_Size;
    Camera m_MainCamera;

    void Start()
    {
        //Fetch the Collider from the GameObject
        m_Collider = GetComponent<Collider>();

        //Fetch the size of the Collider volume
        m_Size = m_Collider.bounds.size;

        //Output to the console the size of the Collider volume

        m_MainCamera = Camera.main;
    }

    public int[] get_bound(int screen_x, int screen_y){

        Vector3[] bounds = {m_Collider.bounds.min, m_Collider.bounds.max};
        float min_x, min_y, max_x, max_y;
        min_x = min_y = 10000;
        max_x = max_y = -10000;
        for(int i = 0; i < 8; i++){
            Vector3 pt = new Vector3(bounds[i / 4].x, bounds[(i % 4) / 2].y, bounds[i % 2].z);
            pt = m_MainCamera.WorldToScreenPoint(pt);
            min_x = Math.Min(min_x, pt.x);
            min_y = Math.Min(min_y, screen_y - pt.y);
            max_x = Math.Max(max_x, pt.x);
            max_y = Math.Max(max_y, screen_y - pt.y);
        }
        int[] return_bound = {(int)Math.Round(min_x), (int)Math.Round(min_y), (int)Math.Round(max_x), (int)Math.Round(max_y)};
        return return_bound;
    }
}
