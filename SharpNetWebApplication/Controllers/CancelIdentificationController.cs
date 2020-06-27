using System;
using Microsoft.AspNetCore.Mvc;
using SharpNetWebApplication.Models;

// For more information on enabling Web API for empty projects, visit https://go.microsoft.com/fwlink/?LinkID=397860

namespace SharpNetWebApplication.Controllers
{
    [Route("[controller]")]
    [ApiController]
    public class CancelIdentificationController : ControllerBase
    {
        [HttpGet]
        public string Get()
        {
            string result = "Cancel Rest API v"+Program.GetCurrentVersion();
            result += " - " + Program.NbRequest + " requests at " +(Program.TotalMilliSecondsForAllRequests / Math.Max(Program.NbRequest,1)) + "ms/request";
            return result;
        }

        [HttpGet("{id}")]
        public CancelIdentification Get(string id)
        {
            return Startup.GetCancelIdentificationIfAvailable(id);
        }

        [HttpPost]
        public void Post(string id)
        {
            Startup.AddComputation(id);
        }
    }
}
