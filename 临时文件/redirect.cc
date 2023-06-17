#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "node.grpc.pb.h"
#include "node.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using node::NodeReply;
using node::NodeRequest;
using node::NodeService;

// Logic and data behind the server's behavior.
class NodeServiceImpl final : public NodeService::Service
{
    Status ForwardRequest(ServerContext *context, const NodeRequest *request,
                          NodeReply *reply) override
    {
        // Determine which node should receive the request based on request data.
        // For simplicity, we assume that this is done by checking a field in the request message.
        int node_number = request->node_number();

        // Send the request to the appropriate node.
        // For simplicity, we assume that the nodes are running on the same machine and can be accessed via localhost.
        std::string target_address = "localhost:" + std::to_string(50000 + node_number);
        std::unique_ptr<NodeService::Stub> stub = NodeService::NewStub(grpc::CreateChannel(
            target_address, grpc::InsecureChannelCredentials()));
        NodeReply node_reply;
        Status status = stub->ForwardRequest(context, *request, &node_reply);
        if (!status.ok())
        {
            std::cout << "Error forwarding request to node: " << status.error_message() << std::endl;
            return status;
        }

        // Return the response to the client.
        reply->set_response(node_reply.response());
        return Status::OK;
    }
};

void RunServer()
{
    std::string server_address("localhost:50001");
    NodeServiceImpl service;

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;
    server->Wait();
}

int main(int argc, char **argv)
{
    RunServer();
    return 0;
}
