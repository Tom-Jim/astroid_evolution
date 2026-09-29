//! WebGPU 异步缓冲区回读与 CPU 结果同步
//! 严格依照 docs/software.md 第 12、14 节 与 ryugu_cauchy_gravity 实现
//!
//! 在 WebGPU/WASM 单线程异步环境下，通过 MapAsync 和 Staging Buffer 安全读取 GPU 计算结果，
//! 避免主线程阻塞或死锁。

/// 从 GPU Staging Buffer 异步读取字节数据
pub async fn readback_staging_buffer_async(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
) -> Result<Vec<u8>, String> {
    let buffer_slice = staging_buffer.slice(..);

    let (sender, receiver) = std::sync::mpsc::channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    // 驱动 GPU 事件循环
    #[cfg(target_arch = "wasm32")]
    {
        let _ = device.poll(wgpu::PollType::Poll);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }

    match receiver.recv() {
        Ok(Ok(())) => {
            let data = buffer_slice
                .get_mapped_range()
                .map_err(|e| format!("{e:?}"))?
                .to_vec();
            staging_buffer.unmap();
            Ok(data)
        }
        Ok(Err(e)) => Err(format!("Buffer map_async 失败: {e:?}")),
        Err(_) => Err("通道接收超时或中断".to_string()),
    }
}

/// 泛型辅助：将回读字节转换为 Pod 切片数据
pub fn cast_readback_slice<T: bytemuck::Pod>(bytes: &[u8]) -> Result<Vec<T>, String> {
    if bytes.len() % std::mem::size_of::<T>() != 0 {
        return Err("回读字节长度与目标数据类型尺寸不对齐".to_string());
    }
    let slice: &[T] = bytemuck::cast_slice(bytes);
    Ok(slice.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cast_readback_slice() {
        let original_data: Vec<f32> = vec![1.0, 2.5, -3.75, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(&original_data);

        let recovered: Vec<f32> = cast_readback_slice(bytes).expect("解析成功");
        assert_eq!(recovered, original_data);
    }
}
