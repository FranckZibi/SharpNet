using System;
using System.Security.Cryptography;

namespace SharpNetWebApplication.Models
{
    /// <summary>
    /// Helper methods for the cancel identification REST API.
    /// </summary>
    public static class CancelIdentificationServices
    {
        /// <summary>
        /// Computes the unique identifier associated with a picture content.
        /// Two pictures with the same content always share the same identifier.
        /// </summary>
        /// <param name="pictureContent">The raw bytes of the picture.</param>
        /// <returns>The SHA-256 hash of the content, as a lower case hexadecimal string.</returns>
        public static string ComputeId(byte[] pictureContent)
        {
            using var sha256 = SHA256.Create();
            var hash = sha256.ComputeHash(pictureContent);
            return Convert.ToHexString(hash).ToLowerInvariant();
        }
    }
}
