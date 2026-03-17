/**
 * Tests for Node.js ONNX inference — tensor serialization/deserialization.
 */

import { describe, test, expect } from '@jest/globals';
import { serializeTensor, deserializeTensor } from '../inference.mjs';

describe('Tensor serialization', () => {
    test('roundtrip preserves data', () => {
        const original = [1.0, 2.0, 3.0, 4.0];
        const shape = [1, 4];

        const serialized = serializeTensor(original, shape);
        const deserialized = deserializeTensor(serialized);

        expect(deserialized.shape).toEqual(shape);
        expect(deserialized.data.length).toBe(4);
        for (let i = 0; i < original.length; i++) {
            expect(deserialized.data[i]).toBeCloseTo(original[i], 5);
        }
    });

    test('serialized format has expected fields', () => {
        const serialized = serializeTensor([1.0, 2.0], [2]);

        expect(serialized).toHaveProperty('data_b64');
        expect(serialized).toHaveProperty('shape');
        expect(serialized).toHaveProperty('dtype');
        expect(serialized.dtype).toBe('float32');
        expect(typeof serialized.data_b64).toBe('string');
    });

    test('handles large tensors', () => {
        const size = 1024;
        const data = Array.from({ length: size }, (_, i) => i * 0.001);
        const shape = [1, size];

        const serialized = serializeTensor(data, shape);
        const deserialized = deserializeTensor(serialized);

        expect(deserialized.data.length).toBe(size);
        expect(deserialized.shape).toEqual(shape);
        expect(deserialized.data[0]).toBeCloseTo(0.0, 5);
        expect(deserialized.data[size - 1]).toBeCloseTo((size - 1) * 0.001, 3);
    });

    test('handles multi-dimensional shapes', () => {
        const data = [1, 2, 3, 4, 5, 6];
        const shape = [2, 3];

        const serialized = serializeTensor(data, shape);
        const deserialized = deserializeTensor(serialized);

        expect(deserialized.shape).toEqual([2, 3]);
        expect(deserialized.data.length).toBe(6);
    });

    test('base64 encoded data is valid', () => {
        const serialized = serializeTensor([1.0, 2.0, 3.0], [3]);

        // Should not throw
        const buf = Buffer.from(serialized.data_b64, 'base64');
        expect(buf.length).toBe(3 * 4); // 3 float32 = 12 bytes
    });

    test('roundtrip with zeros', () => {
        const data = [0.0, 0.0, 0.0];
        const serialized = serializeTensor(data, [3]);
        const deserialized = deserializeTensor(serialized);

        expect(deserialized.data[0]).toBe(0.0);
        expect(deserialized.data[1]).toBe(0.0);
        expect(deserialized.data[2]).toBe(0.0);
    });

    test('roundtrip with negative values', () => {
        const data = [-1.5, 0.0, 1.5];
        const serialized = serializeTensor(data, [3]);
        const deserialized = deserializeTensor(serialized);

        expect(deserialized.data[0]).toBeCloseTo(-1.5, 5);
        expect(deserialized.data[2]).toBeCloseTo(1.5, 5);
    });
});