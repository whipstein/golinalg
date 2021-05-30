package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dgtsv solves the equation
//
//    A*X = B,
//
// where A is an n by n tridiagonal matrix, by Gaussian elimination with
// partial pivoting.
//
// Note that the equation  A**T*X = B  may be solved by interchanging the
// order of the arguments DU and DL.
func Dgtsv(n, nrhs *int, dl, d, du *mat.Vector, b *mat.Matrix, ldb, info *int) {
	var fact, temp, zero float64
	var i, j int

	zero = 0.0

	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
	} else if (*nrhs) < 0 {
		(*info) = -2
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGTSV "), -(*info))
		return
	}

	if (*n) == 0 {
		return
	}

	if (*nrhs) == 1 {
		for i = 1; i <= (*n)-2; i++ {
			if math.Abs(d.Get(i-1)) >= math.Abs(dl.Get(i-1)) {
				//              No row interchange required
				if d.Get(i-1) != zero {
					fact = dl.Get(i-1) / d.Get(i-1)
					d.Set(i+1-1, d.Get(i+1-1)-fact*du.Get(i-1))
					b.Set(i+1-1, 0, b.Get(i+1-1, 0)-fact*b.Get(i-1, 0))
				} else {
					(*info) = i
					return
				}
				dl.Set(i-1, zero)
			} else {
				//              Interchange rows I and I+1
				fact = d.Get(i-1) / dl.Get(i-1)
				d.Set(i-1, dl.Get(i-1))
				temp = d.Get(i + 1 - 1)
				d.Set(i+1-1, du.Get(i-1)-fact*temp)
				dl.Set(i-1, du.Get(i+1-1))
				du.Set(i+1-1, -fact*dl.Get(i-1))
				du.Set(i-1, temp)
				temp = b.Get(i-1, 0)
				b.Set(i-1, 0, b.Get(i+1-1, 0))
				b.Set(i+1-1, 0, temp-fact*b.Get(i+1-1, 0))
			}
		}
		if (*n) > 1 {
			i = (*n) - 1
			if math.Abs(d.Get(i-1)) >= math.Abs(dl.Get(i-1)) {
				if d.Get(i-1) != zero {
					fact = dl.Get(i-1) / d.Get(i-1)
					d.Set(i+1-1, d.Get(i+1-1)-fact*du.Get(i-1))
					b.Set(i+1-1, 0, b.Get(i+1-1, 0)-fact*b.Get(i-1, 0))
				} else {
					(*info) = i
					return
				}
			} else {
				fact = d.Get(i-1) / dl.Get(i-1)
				d.Set(i-1, dl.Get(i-1))
				temp = d.Get(i + 1 - 1)
				d.Set(i+1-1, du.Get(i-1)-fact*temp)
				du.Set(i-1, temp)
				temp = b.Get(i-1, 0)
				b.Set(i-1, 0, b.Get(i+1-1, 0))
				b.Set(i+1-1, 0, temp-fact*b.Get(i+1-1, 0))
			}
		}
		if d.Get((*n)-1) == zero {
			(*info) = (*n)
			return
		}
	} else {
		for i = 1; i <= (*n)-2; i++ {
			if math.Abs(d.Get(i-1)) >= math.Abs(dl.Get(i-1)) {
				//              No row interchange required
				if d.Get(i-1) != zero {
					fact = dl.Get(i-1) / d.Get(i-1)
					d.Set(i+1-1, d.Get(i+1-1)-fact*du.Get(i-1))
					for j = 1; j <= (*nrhs); j++ {
						b.Set(i+1-1, j-1, b.Get(i+1-1, j-1)-fact*b.Get(i-1, j-1))
					}
				} else {
					(*info) = i
					return
				}
				dl.Set(i-1, zero)
			} else {
				//              Interchange rows I and I+1
				fact = d.Get(i-1) / dl.Get(i-1)
				d.Set(i-1, dl.Get(i-1))
				temp = d.Get(i + 1 - 1)
				d.Set(i+1-1, du.Get(i-1)-fact*temp)
				dl.Set(i-1, du.Get(i+1-1))
				du.Set(i+1-1, -fact*dl.Get(i-1))
				du.Set(i-1, temp)
				for j = 1; j <= (*nrhs); j++ {
					temp = b.Get(i-1, j-1)
					b.Set(i-1, j-1, b.Get(i+1-1, j-1))
					b.Set(i+1-1, j-1, temp-fact*b.Get(i+1-1, j-1))
				}
			}
		}
		if (*n) > 1 {
			i = (*n) - 1
			if math.Abs(d.Get(i-1)) >= math.Abs(dl.Get(i-1)) {
				if d.Get(i-1) != zero {
					fact = dl.Get(i-1) / d.Get(i-1)
					d.Set(i+1-1, d.Get(i+1-1)-fact*du.Get(i-1))
					for j = 1; j <= (*nrhs); j++ {
						b.Set(i+1-1, j-1, b.Get(i+1-1, j-1)-fact*b.Get(i-1, j-1))
					}
				} else {
					(*info) = i
					return
				}
			} else {
				fact = d.Get(i-1) / dl.Get(i-1)
				d.Set(i-1, dl.Get(i-1))
				temp = d.Get(i + 1 - 1)
				d.Set(i+1-1, du.Get(i-1)-fact*temp)
				du.Set(i-1, temp)
				for j = 1; j <= (*nrhs); j++ {
					temp = b.Get(i-1, j-1)
					b.Set(i-1, j-1, b.Get(i+1-1, j-1))
					b.Set(i+1-1, j-1, temp-fact*b.Get(i+1-1, j-1))
				}
			}
		}
		if d.Get((*n)-1) == zero {
			(*info) = (*n)
			return
		}
	}

	//     Back solve with the matrix U from the factorization.
	if (*nrhs) <= 2 {
		j = 1
	label70:
		;
		b.Set((*n)-1, j-1, b.Get((*n)-1, j-1)/d.Get((*n)-1))
		if (*n) > 1 {
			b.Set((*n)-1-1, j-1, (b.Get((*n)-1-1, j-1)-du.Get((*n)-1-1)*b.Get((*n)-1, j-1))/d.Get((*n)-1-1))
		}
		for i = (*n) - 2; i >= 1; i-- {
			b.Set(i-1, j-1, (b.Get(i-1, j-1)-du.Get(i-1)*b.Get(i+1-1, j-1)-dl.Get(i-1)*b.Get(i+2-1, j-1))/d.Get(i-1))
		}
		if j < (*nrhs) {
			j = j + 1
			goto label70
		}
	} else {
		for j = 1; j <= (*nrhs); j++ {
			b.Set((*n)-1, j-1, b.Get((*n)-1, j-1)/d.Get((*n)-1))
			if (*n) > 1 {
				b.Set((*n)-1-1, j-1, (b.Get((*n)-1-1, j-1)-du.Get((*n)-1-1)*b.Get((*n)-1, j-1))/d.Get((*n)-1-1))
			}
			for i = (*n) - 2; i >= 1; i-- {
				b.Set(i-1, j-1, (b.Get(i-1, j-1)-du.Get(i-1)*b.Get(i+1-1, j-1)-dl.Get(i-1)*b.Get(i+2-1, j-1))/d.Get(i-1))
			}
		}
	}
}
