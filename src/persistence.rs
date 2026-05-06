//! Implementation on persistence of neural network models

use std::fmt::Display;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::str::FromStr;
use crate::error::*;

pub trait Persistence<U,P,K> where K: PersistenceType {
    /// Load Model
    /// # Arguments
    /// * `persistence` - model persistent object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`ModelLoadError`]
    fn load(&mut self, persistence:&mut P) -> Result<(), ModelLoadError>;
    /// Save Model
    /// # Arguments
    /// * `persistence` - model persistent object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`PersistenceError`]
    fn save(&mut self, persistence:&mut P) -> Result<(), PersistenceError>;
}
pub trait PersistenceType {}
pub struct Specialized;
pub struct Linear;
impl PersistenceType for Specialized {}
impl PersistenceType for Linear {}

/// Trait that defines the implementation of the ability to save a model to a file
pub trait SaveToFile {
    /// Save to File
    /// # Arguments
    /// * `file` - Destination path
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`io::Error`]
    fn save<P: AsRef<Path>>(&self,file:P) -> Result<(),io::Error>;
}
/// A trait that verifies that a read operation on the persistence layer has reached EOF
pub trait VerifyEof {
    /// Has the read position of the persisted information reached EOF?
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`ModelLoadError`]
    fn verify_eof(&mut self) -> Result<(), ModelLoadError>;
}
/// Trait to define an implementation to persist the model in a flat data structure
pub trait LinearPersistence<U> {
    /// Read to restore the persisted model
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`ModelLoadError`]
    fn read(&mut self) -> Result<U, ModelLoadError>;
    /// Write to persist model information
    /// # Arguments
    /// * `u` - Weight value
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`PersistenceError`]
    fn write(&mut self, u:U) -> Result<(), PersistenceError>;
}
/// Types for passing identifiable information about layers and unit boundaries when persisting models
pub enum UnitOrMarker<U> {
    /// Not a boundary.
    Unit(U),
    /// layer boundary
    LayerStart,
    /// boundary
    UnitsStart
}
/// Record type for saving models in text format
pub enum TextRecord {
    F32(f32),
    F64(f64),
    U64(u64),
    LayerStart,
    UnitsStart
}
impl From<f32> for TextRecord {
    fn from(f:f32) -> Self {
        TextRecord::F32(f)
    }
}
impl From<f64> for TextRecord {
    fn from(f:f64) -> Self {
        TextRecord::F64(f)
    }
}
impl From<u64> for TextRecord {
    fn from(i:u64) -> Self {
        TextRecord::U64(i)
    }
}
/// A feature that defines an implementation for persisting a model to a text-based data structure
pub trait TextPersistence<U> {
    /// Read to restore the persisted model
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`ModelLoadError`]
    fn read(&mut self) -> Result<U, ModelLoadError>;
    /// Write to persist model information
    /// # Arguments
    /// * `u` - Weight value
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`PersistenceError`]
    fn write(&mut self, u:UnitOrMarker<U>);
}
/// Persistent object for saving to a text file
pub struct TextFilePersistence {
    reader:Option<BufReader<File>>,
    line:Option<Vec<String>>,
    index:usize,
    data:Vec<TextRecord>
}
impl TextFilePersistence {
    /// Create an instance of TextFilePersistence
    /// # Arguments
    /// * `file` - File path to be persisted
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`ModelLoadError`]
    pub fn new<P: AsRef<Path>>(file:P) -> Result<TextFilePersistence, ModelLoadError> {
        if file.as_ref().exists() {
            Ok(TextFilePersistence {
                reader:Some(BufReader::new(OpenOptions::new().read(true).create(false).open(file)?)),
                line: None,
                index: 0usize,
                data: Vec::new()
            })
        } else {
            Ok(TextFilePersistence {
                reader:None,
                line: None,
                index: 0usize,
                data: Vec::new()
            })
        }
    }

    fn read_line(&mut self) -> Result<String, ModelLoadError> {
        match self.reader {
            Some(ref mut reader) => {
                let mut buf = String::new();
                let n = reader.read_line(&mut buf)?;

                buf = buf.trim().to_string();

                if n == 0 {
                    Err(ModelLoadError::InvalidState(String::from(
                        "End of input has been reached.")))
                } else {
                    Ok(buf)
                }
            },
            None => {
                Err(ModelLoadError::InvalidState(String::from(
                    "File does not exist yet.")))
            }
        }
    }

    fn next_token(&mut self) -> Result<String, ModelLoadError> {
        let t = match self.line {
            None => {
                self.index = 0;
                let mut buf = self.read_line()?;

                while match &*buf {
                    "" => true,
                    s => match s.chars().nth(0) {
                        Some('#') => true,
                        _ => false,
                    }
                } {
                    buf = self.read_line()?;
                }

                let line = buf.split(" ").map(|s| s.to_string()).collect::<Vec<String>>();
                let t = (&line[self.index]).clone();
                self.line = Some(line);
                t
            },
            Some(ref line) => {
                (&line[self.index]).clone()
            }
        };

        self.index = self.index + 1;

        if match self.line {
            Some(ref line) if self.index >= line.len() => {
                true
            },
            Some(_) => {
                false
            }
            None => false,
        } {
            self.line = None;
        }

        Ok(t)
    }
}
impl<U> TextPersistence<U> for TextFilePersistence
    where U: FromStr + Sized,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err>
{
    fn read(&mut self) -> Result<U, ModelLoadError> {
        Ok(self.next_token()?.parse::<U>()?)
    }
    fn write(&mut self, v: UnitOrMarker<U>) {
        match v {
            UnitOrMarker::Unit(u) => {
                self.data.push(u.into());
            },
            UnitOrMarker::LayerStart => {
                self.data.push(TextRecord::LayerStart);
            },
            UnitOrMarker::UnitsStart => {
                self.data.push(TextRecord::UnitsStart);
            }
        }
    }
}
impl VerifyEof for TextFilePersistence {
    fn verify_eof(&mut self) -> Result<(), ModelLoadError> {
        match self.reader {
            Some(ref mut reader) => {
                let mut buf = String::new();

                loop {
                    let n = reader.read_line(&mut buf)?;

                    if n == 0 {
                        return Ok(());
                    }

                    buf = buf.trim().to_string();

                    if !buf.is_empty() {
                        return Err(ModelLoadError::InvalidState(
                            String::from("Data loaded , but the input has not reached the end.")));
                    } else {
                        buf.clear();
                    }
                }
            },
            None => {
                Err(ModelLoadError::InvalidState(String::from(
                    "File does not exist yet.")))
            }
        }
    }
}
impl SaveToFile for TextFilePersistence {
    fn save<P: AsRef<Path>>(&self,file:P) -> Result<(),io::Error> {
        let mut bw = BufWriter::new(OpenOptions::new().write(true).create(true).open(file)?);

        for u in self.data.iter() {
            match u {
                TextRecord::F32(u) => {
                    bw.write(format!("{} ",u).as_bytes())?;
                },
                TextRecord::F64(u) => {
                    bw.write(format!("{} ",u).as_bytes())?;
                },
                TextRecord::U64(u) => {
                    bw.write(format!("{} ",u).as_bytes())?;
                },
                TextRecord::LayerStart => {
                    bw.write(b"#layer\n")?;
                },
                TextRecord::UnitsStart => {
                    bw.write(b"\n")?;
                }
            }
        }

        Ok(())
    }
}
/// Trait that defines a Persistence implementation
/// that stores and loads in fixed length record format.
pub struct BinFilePersistence<U> {
    reader:Option<BufReader<File>>,
    data:Vec<U>
}
impl<U> BinFilePersistence<U> {
    /// Create an instance of TextFilePersistence
    /// # Arguments
    /// * `file` - File path to be persisted
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`ModelLoadError`]
    pub fn new<P: AsRef<Path>>(file:P) -> Result<BinFilePersistence<U>, ModelLoadError> {
        if file.as_ref().exists() {
            Ok(BinFilePersistence {
                reader:Some(BufReader::new(OpenOptions::new().read(true).create(false).open(file)?)),
                data:Vec::new()
            })
        } else {
            Ok(BinFilePersistence {
                reader:None,
                data:Vec::new()
            })
        }
    }
}
impl LinearPersistence<f64> for BinFilePersistence<f64> {
    fn read(&mut self) -> Result<f64, ModelLoadError> {
        match self.reader {
            Some(ref mut reader) => {
                let mut buf = [0; 8];

                reader.read_exact(&mut buf)?;

                Ok(f64::from_bits(
                    (buf[0] as u64) << 56 |
                        (buf[1] as u64) << 48 |
                        (buf[2] as u64) << 40 |
                        (buf[3] as u64) << 32 |
                        (buf[4] as u64) << 24 |
                        (buf[5] as u64) << 16 |
                        (buf[6] as u64) << 8 |
                        buf[7] as u64)
                )
            },
            None => {
                Err(ModelLoadError::InvalidState(String::from(
                    "File does not exist yet.")))
            }
        }
    }

    fn write(&mut self, u: f64) -> Result<(), PersistenceError> {
        self.data.push(u);
        Ok(())
    }
}
impl LinearPersistence<f32> for BinFilePersistence<f32> {
    fn read(&mut self) -> Result<f32, ModelLoadError> {
        match self.reader {
            Some(ref mut reader) => {
                let mut buf = [0; 4];

                reader.read_exact(&mut buf)?;

                Ok(f32::from_bits(
                    (buf[0] as u32) << 24 |
                        (buf[1] as u32) << 16 |
                        (buf[2] as u32) << 8 |
                        buf[3] as u32)
                )
            },
            None => {
                Err(ModelLoadError::InvalidState(String::from(
                    "File does not exist yet.")))
            }
        }
    }

    fn write(&mut self, u: f32) -> Result<(), PersistenceError> {
        self.data.push(u);
        Ok(())
    }
}
impl<U> VerifyEof for BinFilePersistence<U> {
    fn verify_eof(&mut self) -> Result<(), ModelLoadError> {
        match self.reader {
            Some(ref mut reader) => {
                let mut buf: [u8; 1] = [0];

                let n = reader.read(&mut buf)?;

                if n == 0 {
                    Ok(())
                } else {
                    Err(ModelLoadError::InvalidState(String::from("Data loaded , but the input has not reached the end.")))
                }
            },
            None => {
                Err(ModelLoadError::InvalidState(String::from(
                    "File does not exist yet.")))
            }
        }
    }
}
impl SaveToFile for BinFilePersistence<f64> {
    fn save<P: AsRef<Path>>(&self,file:P) -> Result<(),io::Error> {
        let mut bw = BufWriter::new(OpenOptions::new().write(true).create(true).open(file)?);

        for u in self.data.iter() {
            let mut buf = [0; 8];
            let bits = u.to_bits();

            buf[0] = (bits >> 56 & 0xff) as u8;
            buf[1] = (bits >> 48 & 0xff) as u8;
            buf[2] = (bits >> 40 & 0xff) as u8;
            buf[3] = (bits >> 32 & 0xff) as u8;
            buf[4] = (bits >> 24 & 0xff) as u8;
            buf[5] = (bits >> 16 & 0xff) as u8;
            buf[6] = (bits >> 8 & 0xff) as u8;
            buf[7] = (bits & 0xff) as u8;

            bw.write(&buf)?;
        }

        Ok(())
    }
}
impl SaveToFile for BinFilePersistence<f32> {
    fn save<P: AsRef<Path>>(&self,file:P) -> Result<(),io::Error> {
        let mut bw = BufWriter::new(OpenOptions::new().write(true).create(true).open(file)?);

        for u in self.data.iter() {
            let mut buf = [0; 4];
            let bits = u.to_bits();
            buf[0] = (bits >> 24 & 0xff) as u8;
            buf[1] = (bits >> 16 & 0xff) as u8;
            buf[2] = (bits >> 8 & 0xff) as u8;
            buf[3] = (bits & 0xff) as u8;

            bw.write(&buf)?;
        }

        Ok(())
    }
}
