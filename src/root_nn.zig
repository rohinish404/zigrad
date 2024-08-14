const std = @import("std");
const testing = std.testing;
const Engine = @import("engine.zig");
const Value = Engine.Value;
const Neuron = @import("nn.zig").Neuron;
const Layer = @import("nn.zig").Layer;
const MLP = @import("nn.zig").MLP;

test "Neuron initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) @panic("Memory leak detected");
    }
    const allocator = gpa.allocator();

    var neuron = try Neuron.init(allocator, 3);
    defer neuron.deinit();

    try testing.expectEqual(@as(usize, 3), neuron.weights.items.len);
    try testing.expect(neuron.bias.data != 0);
}

test "Neuron call" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) @panic("Memory leak detected");
    }
    const allocator = gpa.allocator();

    var neuron = try Neuron.init(allocator, 2);
    defer neuron.deinit();

    var input = [_]*Value{
        try Value.init(allocator, 1),
        try Value.init(allocator, 2),
    };
    defer {
        input[0].deinit();
        input[1].deinit();
    }

    var result = try neuron.call(&input);
    defer result.deinit();

    try testing.expect(result.data >= 0);  // ReLU activation should always be non-negative
}

test "Layer initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) @panic("Memory leak detected");
    }
    const allocator = gpa.allocator();

    var layer = try Layer.init(allocator, 3, 2);
    defer layer.deinit();

    try testing.expectEqual(@as(usize, 2), layer.neurons.items.len);
    for (layer.neurons.items) |neuron| {
        try testing.expectEqual(@as(usize, 3), neuron.weights.items.len);
    }
}

test "Layer call" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) @panic("Memory leak detected");
    }
    const allocator = gpa.allocator();

    var layer = try Layer.init(allocator, 2, 3);
    defer layer.deinit();

    var input = [_]*Value{
        try Value.init(allocator, 1),
        try Value.init(allocator, 2),
    };
    defer {
        input[0].deinit();
        input[1].deinit();
    }

    var result = try layer.call(&input);
    defer {
        for (result.items) |item| {
            item.deinit();
        }
        result.deinit();
    }

    try testing.expectEqual(@as(usize, 3), result.items.len);
}

test "MLP initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) @panic("Memory leak detected");
    }
    const allocator = gpa.allocator();

    var mlp = try MLP.init(allocator, 2, &[_]u64{3, 1});
    defer mlp.deinit();

    try testing.expectEqual(@as(usize, 3), mlp.sz.items.len);
    try testing.expectEqual(@as(usize, 2), mlp.layers.items.len);
}

test "MLP call" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) @panic("Memory leak detected");
    }
    const allocator = gpa.allocator();

    var mlp = try MLP.init(allocator, 2, &[_]u64{3, 1});
    defer mlp.deinit();

    var input = [_]f64{1, 2};
    var result = try mlp.call(&input);
    defer {
        for (result.items) |item| {
            item.deinit();
        }
        result.deinit();
    }

    try testing.expectEqual(@as(usize, 1), result.items.len);
}

test "MLP parameters" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) @panic("Memory leak detected");
    }
    const allocator = gpa.allocator();

    var mlp = try MLP.init(allocator, 2, &[_]u64{3, 1});
    defer mlp.deinit();

    var params = try mlp.parameters();
    defer params.deinit();

    // Expected number of parameters:
    // Layer 1: (2 inputs * 3 neurons) + 3 biases = 9
    // Layer 2: (3 inputs * 1 neuron) + 1 bias = 4
    // Total: 9 + 4 = 13
    try testing.expectEqual(@as(usize, 13), params.items.len);
}
