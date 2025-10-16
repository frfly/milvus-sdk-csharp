using System.Text.Json;
using Xunit;

namespace Milvus.Client.Tests;

[Collection("Milvus")]
public class HybridSearchTests(
    MilvusFixture milvusFixture,
    HybridSearchTests.QueryCollectionFixture queryCollectionFixture)
    : IClassFixture<HybridSearchTests.QueryCollectionFixture>, IDisposable
{
    [Fact]
    public async Task HybridSearch_with_single_request()
    {
        List<HybridSearchRequest<float>> searchRequests =
        [
            HybridSearchRequest.CreateFloat(
                "float_vector",
                new[] { 0.1f, 0.2f },
                SimilarityMetricType.L2,
                2)
        ];

        var ranker = FunctionSchema.RRFRanker("rff");
        var results = await Collection.HybridSearchAsync(searchRequests, ranker, 2);

        Assert.Equal(CollectionName, results.CollectionName);
        Assert.Empty(results.FieldsData);
        Assert.Collection(results.Ids.LongIds!,
            id => Assert.Equal(1, id),
            id => Assert.Equal(2, id));
        Assert.Null(results.Ids.StringIds);
        Assert.Equal(1, results.NumQueries);
        Assert.Equal(2, results.Scores.Count);
        Assert.Equal(2, results.Limit);
        Assert.Collection(results.Limits, l => Assert.Equal(2, l));
    }

    [Fact]
    public async Task HybridSearch_with_multiple_requests()
    {
        List<HybridSearchRequest<float>> searchRequests =
        [
            HybridSearchRequest.CreateFloat(
                "float_vector",
                new[] { 0.1f, 0.2f },
                SimilarityMetricType.L2,
                1),
            HybridSearchRequest.CreateFloat(
                "float_vector",
                new[] { 5.1f, 6.2f },
                SimilarityMetricType.L2,
                1)
        ];

        var ranker = FunctionSchema.RRFRanker("rff");
        var results = await Collection.HybridSearchAsync(searchRequests, ranker, 2);

        Assert.Equal(CollectionName, results.CollectionName);
        Assert.Empty(results.FieldsData);
        Assert.Collection(results.Ids.LongIds!,
            id => Assert.Equal(1, id),
            id => Assert.Equal(3, id));
        Assert.Null(results.Ids.StringIds);
        Assert.Equal(1, results.NumQueries);
        Assert.Equal(2, results.Scores.Count);
        Assert.Equal(2, results.Limit);
        Assert.Collection(results.Limits, l => Assert.Equal(2, l));
    }

    [Fact]
    public async Task HybridSearch_with_multiple_vectors()
    {
        List<HybridSearchRequest<float>> searchRequests =
        [
            HybridSearchRequest.CreateFloat(
                "float_vector",
                new[] { 0.1f, 0.2f },
                SimilarityMetricType.L2,
                1),
            HybridSearchRequest.CreateFloat(
                "float_vector2",
                new[] { 0.1f, 0.2f, 0.3f },
                SimilarityMetricType.L2,
                1)
        ];

        var ranker = FunctionSchema.RRFRanker("rff");
        var results = await Collection.HybridSearchAsync(searchRequests, ranker, 2);

        Assert.Equal(CollectionName, results.CollectionName);
        Assert.Empty(results.FieldsData);
        Assert.Collection(results.Ids.LongIds!,
            id => Assert.Equal(1, id),
            id => Assert.Equal(5, id));
        Assert.Null(results.Ids.StringIds);
        Assert.Equal(1, results.NumQueries);
        Assert.Equal(2, results.Scores.Count);
        Assert.Equal(2, results.Limit);
        Assert.Collection(results.Limits, l => Assert.Equal(2, l));
    }

    [Fact]
    public async Task HybridSearch_with_OutputFields()
    {
        List<HybridSearchRequest<float>> searchRequests =
        [
            HybridSearchRequest.CreateFloat(
                "float_vector",
                new[] { 0.1f, 0.2f },
                SimilarityMetricType.L2,
                1),
            HybridSearchRequest.CreateFloat(
                "float_vector",
                new[] { 5.1f, 6.2f },
                SimilarityMetricType.L2,
                1)
        ];

        var ranker = FunctionSchema.RRFRanker("rff");
        var results = await Collection.HybridSearchAsync(searchRequests, ranker, 2, ["varchar", "json_thing"]);

        Assert.Equal(CollectionName, results.CollectionName);
        Assert.Collection(results.Ids.LongIds!,
            id => Assert.Equal(1, id),
            id => Assert.Equal(3, id));
        Assert.Collection((results.FieldsData.First(fd => fd.FieldName == "varchar") as FieldData<string>)!.Data,
            str => Assert.Equal("one", str),
            str => Assert.Equal("three", str));
        Assert.Collection((results.FieldsData.First(fd => fd.FieldName == "json_thing") as FieldData<string>)!.Data,
            jsonStr => Assert.Equal(1, JsonSerializer.Deserialize<JsonThing>(jsonStr)!.Number),
            jsonStr => Assert.Equal(3, JsonSerializer.Deserialize<JsonThing>(jsonStr)!.Number));
        Assert.Null(results.Ids.StringIds);
        Assert.Equal(1, results.NumQueries);
        Assert.Equal(2, results.Scores.Count);
        Assert.Equal(2, results.Limit);
        Assert.Collection(results.Limits, l => Assert.Equal(2, l));
    }

    [Fact]
    public async Task HybridSearch_with_weighted_rerank()
    {
        List<HybridSearchRequest<float>> searchRequests =
        [
            HybridSearchRequest.CreateFloat(
                "float_vector",
                new[] { 0.1f, 0.2f },
                SimilarityMetricType.L2,
                2),
            HybridSearchRequest.CreateFloat(
                "float_vector",
                new[] { 8f, 9f },
                SimilarityMetricType.L2,
                2),
        ];

        var ranker = FunctionSchema.WeightedRanker("weighted", [0.1f, 0.9f], normScore: true);
        var results = await Collection.HybridSearchAsync(searchRequests, ranker, 2);

        Assert.Equal(CollectionName, results.CollectionName);
        Assert.Empty(results.FieldsData);
        Assert.Collection(results.Ids.LongIds!,
            id => Assert.Equal(5, id),
            id => Assert.Equal(4, id));
        Assert.Null(results.Ids.StringIds);
        Assert.Equal(1, results.NumQueries);
        Assert.Equal(2, results.Scores.Count);
        Assert.Equal(2, results.Limit);
        Assert.Collection(results.Limits, l => Assert.Equal(2, l));
    }

    [Fact]
    public async Task HybridSearch_with_custom_rerank()
    {
        List<HybridSearchRequest<float>> searchRequests =
        [
            HybridSearchRequest.CreateFloat(
                "float_vector",
                new[] { 0.1f, 0.2f },
                SimilarityMetricType.L2,
                5),
            HybridSearchRequest.CreateFloat(
                "float_vector",
                new[] { 8f, 9f },
                SimilarityMetricType.L2,
                5),
        ];

        var ranker = new FunctionSchema()
        {
            Type = FunctionType.Rerank,
            InputFieldNames = ["timestamp"],
            OutputFieldNames = [],
            Name = "decay",
            Params = new Dictionary<string, object>()
            {
                ["reranker"] = "decay",
                ["function"] = "gauss",
                ["origin"] = GetTimestamp(new DateTime(2025, 1, 3)),
                ["scale"] = (int)TimeSpan.FromDays(1).TotalSeconds,
                ["offset"] = (int)TimeSpan.FromDays(1).TotalSeconds,
                ["decay"] = 0.5
            }
        };

        var results = await Collection.HybridSearchAsync(searchRequests, ranker, 2);

        Assert.Equal(CollectionName, results.CollectionName);
        Assert.Empty(results.FieldsData);
        Assert.Collection(results.Ids.LongIds!,
            id => Assert.Equal(4, id),
            id => Assert.Equal(5, id));
        Assert.Null(results.Ids.StringIds);
        Assert.Equal(1, results.NumQueries);
        Assert.Equal(2, results.Scores.Count);
        Assert.Equal(2, results.Limit);
        Assert.Collection(results.Limits, l => Assert.Equal(2, l));
    }

    [Fact]
    public async Task HybridSearch_failed_with_invalid_function()
    {
        List<HybridSearchRequest<float>> searchRequests =
        [
            HybridSearchRequest.CreateFloat(
                "float_vector",
                new[] { 0.1f, 0.2f },
                SimilarityMetricType.L2,
                5),
            HybridSearchRequest.CreateFloat(
                "float_vector",
                new[] { 8f, 9f },
                SimilarityMetricType.L2,
                5),
        ];

        var ranker = new FunctionSchema
        {
            Type = FunctionType.Bm25,
            InputFieldNames = [],
            OutputFieldNames = [],
            Name = "foo",
            Params = null
        };

        await Assert.ThrowsAsync<ArgumentException>(async
            () => await Collection.HybridSearchAsync(searchRequests, ranker, 2));
    }

    public class QueryCollectionFixture : IAsyncLifetime
    {
        public QueryCollectionFixture(MilvusFixture milvusFixture)
        {
            Client = milvusFixture.CreateClient();
            Collection = Client.GetCollection(nameof(SearchQueryTests));
        }

        private readonly MilvusClient Client;
        public readonly MilvusCollection Collection;

        public async Task InitializeAsync()
        {
            await Collection.DropAsync();
            await Client.CreateCollectionAsync(
                Collection.Name,
                [
                    FieldSchema.Create<long>("id", isPrimaryKey: true),
                    FieldSchema.CreateVarchar("varchar", 256),
                    FieldSchema.Create<long>("timestamp"),
                    FieldSchema.CreateJson("json_thing"),
                    FieldSchema.CreateFloatVector("float_vector", 2),
                    FieldSchema.CreateFloatVector("float_vector2", 3),
                ]);

            await Collection.CreateIndexAsync(
                "float_vector", IndexType.Flat, SimilarityMetricType.L2, "float_vector_idx",
                new Dictionary<string, string>());

            await Collection.CreateIndexAsync(
                "float_vector2", IndexType.Flat, SimilarityMetricType.L2, "float_vector2_idx",
                new Dictionary<string, string>());

            long[] ids = [1, 2, 3, 4, 5];
            long[] timestamps =
            [
                GetTimestamp(new DateTime(2025, 01, 01)),
                GetTimestamp(new DateTime(2025, 01, 02)),
                GetTimestamp(new DateTime(2025, 01, 03)),
                GetTimestamp(new DateTime(2025, 01, 04)),
                GetTimestamp(new DateTime(2025, 01, 05))
            ];
            string[] strings = ["one", "two", "three", "four", "five"];
            ReadOnlyMemory<float>[] floatVectors =
            [
                new[] { 1f, 2f },
                new[] { 3.5f, 4.5f },
                new[] { 5f, 6f },
                new[] { 7.7f, 8.8f },
                new[] { 9f, 10f }
            ];

            ReadOnlyMemory<float>[] floatVectors2 =
            [
                new[] { 13f, 14f, 15 },
                new[] { 10f, 11f, 12f },
                new[] { 7f, 8f, 9f },
                new[] { 4f, 5f, 6f },
                new[] { 1f, 2f, 3f }
            ];

            List<string> jsons = Enumerable.Range(1, 5)
                .Select(i => JsonSerializer.Serialize(new JsonThing { Title = "Title" + i, Number = i }))
                .ToList();

            await Collection.InsertAsync(
            [
                FieldData.Create("id", ids),
                FieldData.Create("varchar", strings),
                FieldData.Create("timestamp", timestamps),
                FieldData.CreateJson("json_thing", jsons),
                FieldData.CreateFloatVector("float_vector", floatVectors),
                FieldData.CreateFloatVector("float_vector2", floatVectors2)
            ]);

            await Collection.LoadAsync();
            await Collection.WaitForCollectionLoadAsync(
                waitingInterval: TimeSpan.FromMilliseconds(100), timeout: TimeSpan.FromMinutes(1));
        }

        public Task DisposeAsync()
        {
            Client.Dispose();
            return Task.CompletedTask;
        }
    }

    private readonly MilvusClient Client = milvusFixture.CreateClient();

    private MilvusCollection Collection => queryCollectionFixture.Collection;
    private string CollectionName => Collection.Name;

    public void Dispose() => Client.Dispose();

    internal class JsonThing
    {
        public string? Title { get; set; }
        public int Number { get; set; }
    }

    private static long GetTimestamp(DateTime dateTime)
    {
        return (long)(dateTime.ToUniversalTime() - DateTime.UnixEpoch).TotalSeconds;
    }
}
