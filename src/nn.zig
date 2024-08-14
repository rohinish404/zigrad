const std = @import("std");
const Engine = @import("engine.zig");
const Value = Engine.Value;
const c = @cImport({
    @cInclude("stdlib.h");
    @cInclude("time.h");
});

fn rand_float() f64{
    const random_float:f64 = @as(f64, @floatFromInt(c.rand())) / @as(f64, @floatFromInt(c.RAND_MAX));
    return random_float;
}

pub const Neuron = struct {
    weights: std.ArrayList(*Value),
    bias: *Value,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, nin: u64) !Neuron {
        var weights = std.ArrayList(*Value).init(allocator);
        errdefer weights.deinit();
        const bias = try Value.init(allocator,rand_float());
        errdefer bias.deinit();
        for (0..nin) |_| {
            const random_weight = rand_float();
            const weight = try Value.init(allocator, random_weight);
            errdefer weight.deinit();
            try weights.append(weight);
        }
        return .{
            .weights = weights,
            .allocator = allocator,
            .bias = bias,
        };
    }

    pub fn call(self: *Neuron, x: []*Value) !*Value {
        var result = try Value.init(self.allocator, 0);
        errdefer result.deinit();
        for (self.weights.items, 0..) |weight, i| {
            const product = try weight.mult(x[i]);
            defer product.deinit();
            const temp_result = try result.add(product);
            result.deinit();
            result = temp_result;
        }
        const temp_result = try result.add(self.bias);
        result.deinit(); 
        result = temp_result;
        const final_result = try result.relu();
        result.deinit();
        result = final_result;
        return result;
    }

    pub fn parameters(self: *Neuron) !std.ArrayList(*Value){
        var result = std.ArrayList(*Value).init(self.allocator);
        errdefer result.deinit();
        try result.appendSlice(self.weights.items);
        try result.append(self.bias);
        return result;
    }

    pub fn deinit(self: *Neuron) void {

        for (self.weights.items) |weight| {
            weight.deinit();
        }
        self.weights.deinit();
        self.bias.deinit();
    }


};

pub const Layer = struct {
    neurons: std.ArrayList(Neuron),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, nin: u64, nout: u64) !Layer {
        var neurons = std.ArrayList(Neuron).init(allocator);
        errdefer neurons.deinit();
        for (0..nout) |_| {
            var neuron = try Neuron.init(allocator, nin);
            errdefer neuron.deinit();
            try neurons.append(neuron);
        }
        return .{
            .neurons = neurons,
            .allocator = allocator,
        };
    }

    pub fn call(self: *Layer, x: []*Value) !std.ArrayList(*Value) {
        var final = std.ArrayList(*Value).init(self.allocator);
        errdefer {
            for (final.items) |item| {
                item.deinit();
            }
            final.deinit();
        }
        for (self.neurons.items) |*neuron| {
            const result = try neuron.call(x);
            errdefer result.deinit();
            try final.append(result);
        }
        return final;
    }

    pub fn parameters(self: *Layer) !std.ArrayList(*Value){
        var result = std.ArrayList(*Value).init(self.allocator);
        errdefer result.deinit();
        for (self.neurons.items) |*neuron|{
            var neuro_params = try neuron.parameters();
            defer neuro_params.deinit();
            try result.appendSlice(neuro_params.items);
        }
        return result;
    }


    pub fn deinit(self: *Layer) void {
        for (self.neurons.items) |*neuron| {
            neuron.deinit();
        }
        self.neurons.deinit();
    }
};

pub const MLP = struct {
    sz: std.ArrayList(u64),
    layers: std.ArrayList(Layer),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, nin:u64, nout: []const u64) !MLP {
        var sz = std.ArrayList(u64).init(allocator);
        errdefer sz.deinit();
        try sz.append(nin);
        for (nout) |out| {
            try sz.append(out);
        }
        var layers = std.ArrayList(Layer).init(allocator);
        errdefer {
            for (layers.items) |*layer| {
                layer.deinit();
            }
            layers.deinit();
        }
        for (nout, 0..) |_,i| {
            var layer = try Layer.init(allocator, sz.items[i], sz.items[i+1]);
            errdefer layer.deinit();
            try layers.append(layer);
        }
        return .{
            .sz=sz,
            .layers=layers,
            .allocator=allocator,
        };

    }
    pub fn call(self: *MLP, x: []const f64) !std.ArrayList(*Value) {
        var input = try std.ArrayList(*Value).initCapacity(self.allocator, x.len);
        errdefer {
            for (input.items) |val| {
                val.deinit();
            }
            input.deinit();
        }

        for (x) |val| {
            const value = try Value.init(self.allocator, val);
            errdefer value.deinit();
            try input.append(value);
        }

        for (self.layers.items) |*layer| {
            const new_input = try layer.call(input.items);
            for (input.items) |val| {
                val.deinit();
            }
            input.deinit();
            input = new_input;
        }

        return input;
    }
    pub fn parameters(self: *MLP) !std.ArrayList(*Value) {
        var result = std.ArrayList(*Value).init(self.allocator);
        errdefer result.deinit();

        for (self.layers.items) |*layer| {
            var layer_params = try layer.parameters();
            defer layer_params.deinit();

            try result.appendSlice(layer_params.items);
        }

        return result;
    }

    pub fn deinit(self: *MLP) void {
        for (self.layers.items) |*layer| {
            layer.deinit();
        }
        self.sz.deinit();
        self.layers.deinit();
    }

};
