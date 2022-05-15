extern crate cc;

use std::env;

fn find_library_paths() -> Vec<String> {
	match env::var("CUDA_PATH") {
		Ok(path) => {
			let split_char = if cfg!(target_os = "windows") { ";" } else { ":" };

			path.split(split_char).map(|s| s.to_owned()).collect::<Vec<_>>()
		}
		Err(_) => vec![],
	}
}

fn main() {
	println!("cargo:rerun-if-changed=src/kernel.cu");
	
	let library_paths = find_library_paths();

	let out_dir = env::var("OUT_DIR").unwrap();

	cc::Build::new()
		.cuda(true)
		.flag("-cudart=shared")
		.flag("-gencode")
		.flag("arch=compute_61,code=sm_61")
		.file("src/kernel.cu")
		.compile("libkernel.a");

	for p in library_paths {
		println!("cargo:rustc-link-search=native={}/lib/x64", p);
	}
	println!("cargo:rustc-link-lib=cudart");
	println!("cargo:rustc-link-search=native={}", out_dir);
	println!("cargo:rustc-link-lib=static=kernel");
}
