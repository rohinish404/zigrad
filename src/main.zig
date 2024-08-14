const std = @import("std");
const math =  std.math;
const c = @cImport({
    @cInclude("stdlib.h");
    @cInclude("time.h");
});
const Engine = @import("engine.zig");
const Value = Engine.Value;
const NN = @import("nn.zig");
const MLP = NN.MLP;


pub fn main() !void {
    c.srand(@as(c_uint, @intCast(c.time(null))));
    std.debug.print("Hello world\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    // inputs
    //var x1 = try Value.init(2.0);
    //var x2 =  try Value.init(0.0);
    ////weights
    //var w1 = try Value.init(-3.0);
    //var w2 = try Value.init(1.0);
    ////bias
    //var b = try Value.init(6.7);

    //var x1w1 = try x1.mult(&w1);
    //var x2w2 = try x2.mult(&w2);
    //var x1w1x2w2 = try x1w1.add(&x2w2);

    //var n = try x1w1x2w2.add(&b);

    //var o = try n.relu();

    ////std.debug.print("data = {d:.2}, grad = {d:.2}, children = {any}\n", .{o.data, o.grad, o.children}); 
    ////std.debug.print("-----------------------\n", .{});

    //try o.backward(allocator);
    //std.debug.print("data = {d:.2}, grad = {d:.2}, children = {any}\n", .{n.data, n.grad, n.children});

    std.debug.print("------------------------------------------------------------\n", .{});
    var arr = [_]u64{4,4,1};
    var mlp = try MLP.init(allocator, 3, arr[0..]);
    defer mlp.deinit();


    // var arr2 = [_]f64{2.0,3.0, -1.0};
    // const res = try mlp.call(arr2[0..]);
    // defer res.deinit();
    // std.debug.print("res - {d:.2}\n", .{res.items[0].data});

    // const res_p = try mlp.parameters();
    // defer res_p.deinit();
    // std.debug.print("res_p - {}\n", .{res_p.items.len});


    const input = [_][3]f64{
    [_]f64{2.0, 3.0, -1.0},
    [_]f64{3.0, -1.0, 0.5},
    [_]f64{0.5, 1.0, 1.0},
    [_]f64{1.0, 1.0, -1.0},
    };

    const output = [_]f64{1.0, -1.0, -1.0, 1.0};
    var val_output = std.ArrayList(*Value).init(allocator);
    defer {
        for (val_output.items) |value| {
            value.deinit();
        }
        val_output.deinit();

    }
    for (output) |op| {
        const temp_val = try Value.init(allocator, op);
        std.debug.print("temp-val - {any}\n", .{temp_val});

        try val_output.append(temp_val);
    }

    var y_pred = std.ArrayList(std.ArrayList(*Value)).init(allocator);
    defer {
        for (y_pred.items) |item| {
            for (item.items) |value| {
                value.deinit();
            }
            item.deinit();
        }
        y_pred.deinit();
    }

    for (input) |in| {
        const res = try mlp.call(@constCast(&in));
        try y_pred.append(res);

    }
    std.debug.print("y_pred - {any}\n", .{y_pred.items.len});
    var loss = try Value.init(allocator, 0.0);
    defer loss.deinit();
    
    var total_loss = try Value.init(allocator, 0.0);
    defer total_loss.deinit();
    
    std.debug.print("val_output - {any}\n", .{val_output.items});
    for (y_pred.items, 0..) |pred, i| {
        std.debug.print("pred length - {}", .{y_pred.items.len});
        std.debug.print("Processing sample {}: pred={d:.6}, actual={d:.6}\n", .{ i, pred.items[0].data, val_output.items[i].data });

        const ans = try pred.items[0].sub(val_output.items[i]);
        std.debug.print("  Difference (pred - actual): {d:.6}\n", .{ans.data});

        const sq_ans = try ans.pow(2.0);
        std.debug.print("  Squared difference: {d:.6}\n", .{sq_ans.data});
        
        const temp_loss = try total_loss.add(sq_ans);
        total_loss = temp_loss;
        
        std.debug.print("  Updated loss: {d:.6}\n", .{total_loss.data});
        
        if (total_loss.children) |children| {
            std.debug.print("  Loss children: self={*}, other={*}\n", .{children.self, children.other});
            std.debug.print("  Self data: {d:.6}, Other data: {d:.6}\n", .{children.self.data, children.other.?.data});
        } else {
            std.debug.print("  Loss has no children\n", .{});
        }
    }
    loss.deinit(); 
    loss = total_loss;
    
    std.debug.print("loss------------------{any}\n", .{loss});
    if (loss.children) |children| {
        std.debug.print("Final loss children: self={*}, other={*}\n", .{children.self, children.other});
        std.debug.print("Self data: {d:.6}, Other data: {d:.6}\n", .{children.self.data, children.other.?.data});
    } else {
        std.debug.print("Final loss has no children\n", .{});
    }

    std.debug.print("Loss Variable Information:\n", .{});
    std.debug.print("  Data: {d:.6}\n", .{loss.data});

    std.debug.print("  Gradient: {d:.6}\n", .{loss.grad});

    //try loss.backward();


}




