
# Brief description of the structure of this project


This project represents a simple test than can perform RDMA to a remote GPU. All the source files are inside the `plugins` directory. 

  - `NumberProducer.cc`: Produces an std::vector of random numbers and puts them on the event.
  - `NumberOffloader.cc`: Send those numbers to a remote CPU or GPU (there are flags to specify this in the code.)
  - `NumberAccS.cc`: The "accumulator" on the server (remote machine). It sums all the numbers in the std::vector, and outputs another std::vector consisting only on one element.
  - `cudaWrapper.cu`: Contains the CUDA kernel that actually performs the sum. It is called from `NumberAccS`.
  - `NumberLogger.cc`: Takes an std::vector as input and logs on the terminal all of its elements.
  - `NumberAccumulator.cc`: In case we don't want to perform the computation in a remote machine, this script does it on the local host. We can use either this one or `NumberAccS.cc`, by specifying it on the Python configuration files.

The Python configuration files are under the `test` folder. There is one for the client and another one for the server.









