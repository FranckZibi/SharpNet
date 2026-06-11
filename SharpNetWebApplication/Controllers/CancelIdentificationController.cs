using System;
using System.IO;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using SharpNetWebApplication.Models;

// For more information on enabling Web API for empty projects, visit https://go.microsoft.com/fwlink/?LinkID=397860

namespace SharpNetWebApplication.Controllers
{
    /// <summary>
    /// REST API to identify the cancel of a stamp picture.
    /// The client uploads the content of the picture (not a path) and retrieves the result with the returned id.
    /// </summary>
    /// <remarks>
    /// Not unit tested: this controller is a thin HTTP layer over <see cref="Startup"/> (which loads a trained
    /// neural network in a static computation thread). The pure id computation is tested behind
    /// <see cref="CancelIdentificationServices.ComputeId"/>.
    /// </remarks>
    [Route("[controller]")]
    [ApiController]
    public class CancelIdentificationController : ControllerBase
    {
        /// <summary>
        /// Gets the version of the REST API and some usage statistics.
        /// </summary>
        /// <returns>A human readable status string.</returns>
        [HttpGet]
        public string Get()
        {
            string result = "Cancel Rest API v"+Program.GetCurrentVersion();
            result += " - " + Program.NbRequest + " requests at " +(Program.TotalMilliSecondsForAllRequests / Math.Max(Program.NbRequest,1)) + "ms/request";
            return result;
        }

        /// <summary>
        /// Gets the cancel identification associated with the provided id (the hash of the picture content,
        /// as returned by the POST method).
        /// </summary>
        /// <param name="id">The id of the cancel identification to retrieve.</param>
        /// <returns>The cancel identification (its <see cref="CancelIdentification.IsDone"/> flag tells if the prediction is available).</returns>
        [HttpGet("{id}")]
        public CancelIdentification Get(string id)
        {
            return Startup.GetCancelIdentificationIfAvailable(id);
        }

        /// <summary>
        /// Starts the cancel identification of the provided picture content.
        /// </summary>
        /// <param name="picture">The picture to identify, uploaded as a multipart/form-data file.</param>
        /// <returns>
        /// The (possibly not yet done) cancel identification: its <see cref="CancelIdentification.Id"/>
        /// (the hash of the picture content) is to be used to retrieve the result with the GET method.
        /// </returns>
        [HttpPost]
        public CancelIdentification Post(IFormFile picture)
        {
            if (picture == null)
            {
                return Startup.AddComputation(Array.Empty<byte>());
            }
            using var pictureContent = new MemoryStream();
            picture.CopyTo(pictureContent);
            return Startup.AddComputation(pictureContent.ToArray());
        }
    }
}
