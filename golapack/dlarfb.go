package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlarfb applies a real block reflector H or its transpose H**T to a
// real m by n matrix C, from either the left or the right.
func Dlarfb(side mat.MatSide, trans mat.MatTrans, direct, storev byte, m, n, k int, v, t, c, work *mat.Matrix) {
	var transt mat.MatTrans
	var one float64
	var i, j int
	var err error

	one = 1.0

	//     Quick return if possible
	if m <= 0 || n <= 0 {
		return
	}

	if trans == NoTrans {
		transt = Trans
	} else {
		transt = NoTrans
	}

	if storev == 'C' {
		if direct == 'F' {
			//           Let  V =  ( V1 )    (first K rows)
			//                     ( V2 )
			//           where  V1  is unit lower triangular.
			if side == Left {
				//              Form  H * C  or  H**T * C  where  C = ( C1 )
				//                                                    ( C2 )
				//
				//              W := C**T * V  =  (C1**T * V1 + C2**T * V2)  (stored in WORK)
				//
				//              W := C1**T
				for j = 1; j <= k; j++ {
					goblas.Dcopy(n, c.Vector(j-1, 0), work.Vector(0, j-1, 1))
				}

				//              W := W * V1
				if err = goblas.Dtrmm(Right, Lower, NoTrans, Unit, n, k, one, v, work); err != nil {
					panic(err)
				}
				if m > k {
					//                 W := W + C2**T * V2
					if err = goblas.Dgemm(Trans, NoTrans, n, k, m-k, one, c.Off(k, 0), v.Off(k, 0), one, work); err != nil {
						panic(err)
					}
				}

				//              W := W * T**T  or  W * T
				if err = goblas.Dtrmm(Right, Upper, transt, NonUnit, n, k, one, t, work); err != nil {
					panic(err)
				}

				//              C := C - V * W**T
				if m > k {
					//                 C2 := C2 - V2 * W**T
					if err = goblas.Dgemm(NoTrans, Trans, m-k, n, k, -one, v.Off(k, 0), work, one, c.Off(k, 0)); err != nil {
						panic(err)
					}
				}

				//              W := W * V1**T
				if err = goblas.Dtrmm(Right, Lower, Trans, Unit, n, k, one, v, work); err != nil {
					panic(err)
				}

				//              C1 := C1 - W**T
				for j = 1; j <= k; j++ {
					for i = 1; i <= n; i++ {
						c.Set(j-1, i-1, c.Get(j-1, i-1)-work.Get(i-1, j-1))
					}
				}

			} else if side == Right {
				//              Form  C * H  or  C * H**T  where  C = ( C1  C2 )
				//
				//              W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)
				//
				//              W := C1
				for j = 1; j <= k; j++ {
					goblas.Dcopy(m, c.Vector(0, j-1, 1), work.Vector(0, j-1, 1))
				}

				//              W := W * V1
				if err = goblas.Dtrmm(Right, Lower, NoTrans, Unit, m, k, one, v, work); err != nil {
					panic(err)
				}
				if n > k {
					//                 W := W + C2 * V2
					if err = goblas.Dgemm(NoTrans, NoTrans, m, k, n-k, one, c.Off(0, k), v.Off(k, 0), one, work); err != nil {
						panic(err)
					}
				}

				//              W := W * T  or  W * T**T
				if err = goblas.Dtrmm(Right, Upper, trans, NonUnit, m, k, one, t, work); err != nil {
					panic(err)
				}

				//              C := C - W * V**T
				if n > k {
					//                 C2 := C2 - W * V2**T
					if err = goblas.Dgemm(NoTrans, Trans, m, n-k, k, -one, work, v.Off(k, 0), one, c.Off(0, k)); err != nil {
						panic(err)
					}
				}

				//              W := W * V1**T
				if err = goblas.Dtrmm(Right, Lower, Trans, Unit, m, k, one, v, work); err != nil {
					panic(err)
				}

				//              C1 := C1 - W
				for j = 1; j <= k; j++ {
					for i = 1; i <= m; i++ {
						c.Set(i-1, j-1, c.Get(i-1, j-1)-work.Get(i-1, j-1))
					}
				}
			}

		} else {
			//           Let  V =  ( V1 )
			//                     ( V2 )    (last K rows)
			//           where  V2  is unit upper triangular.
			if side == Left {
				//              Form  H * C  or  H**T * C  where  C = ( C1 )
				//                                                    ( C2 )
				//
				//              W := C**T * V  =  (C1**T * V1 + C2**T * V2)  (stored in WORK)
				//
				//              W := C2**T
				for j = 1; j <= k; j++ {
					goblas.Dcopy(n, c.Vector(m-k+j-1, 0), work.Vector(0, j-1, 1))
				}

				//              W := W * V2
				if err = goblas.Dtrmm(Right, Upper, NoTrans, Unit, n, k, one, v.Off(m-k, 0), work); err != nil {
					panic(err)
				}
				if m > k {
					//                 W := W + C1**T * V1
					if err = goblas.Dgemm(Trans, NoTrans, n, k, m-k, one, c, v, one, work); err != nil {
						panic(err)
					}
				}

				//              W := W * T**T  or  W * T
				if err = goblas.Dtrmm(Right, Lower, transt, NonUnit, n, k, one, t, work); err != nil {
					panic(err)
				}

				//              C := C - V * W**T
				if m > k {
					//                 C1 := C1 - V1 * W**T
					if err = goblas.Dgemm(NoTrans, Trans, m-k, n, k, -one, v, work, one, c); err != nil {
						panic(err)
					}
				}

				//              W := W * V2**T
				if err = goblas.Dtrmm(Right, Upper, Trans, Unit, n, k, one, v.Off(m-k, 0), work); err != nil {
					panic(err)
				}

				//              C2 := C2 - W**T
				for j = 1; j <= k; j++ {
					for i = 1; i <= n; i++ {
						c.Set(m-k+j-1, i-1, c.Get(m-k+j-1, i-1)-work.Get(i-1, j-1))
					}
				}

			} else if side == Right {
				//              Form  C * H  or  C * H**T  where  C = ( C1  C2 )
				//
				//              W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)
				//
				//              W := C2
				for j = 1; j <= k; j++ {
					goblas.Dcopy(m, c.Vector(0, n-k+j-1, 1), work.Vector(0, j-1, 1))
				}

				//              W := W * V2
				if err = goblas.Dtrmm(Right, Upper, NoTrans, Unit, m, k, one, v.Off(n-k, 0), work); err != nil {
					panic(err)
				}
				if n > k {
					//                 W := W + C1 * V1
					if err = goblas.Dgemm(NoTrans, NoTrans, m, k, n-k, one, c, v, one, work); err != nil {
						panic(err)
					}
				}

				//              W := W * T  or  W * T**T
				if err = goblas.Dtrmm(Right, Lower, trans, NonUnit, m, k, one, t, work); err != nil {
					panic(err)
				}

				//              C := C - W * V**T
				if n > k {
					//                 C1 := C1 - W * V1**T
					if err = goblas.Dgemm(NoTrans, Trans, m, n-k, k, -one, work, v, one, c); err != nil {
						panic(err)
					}
				}

				//              W := W * V2**T
				if err = goblas.Dtrmm(Right, Upper, Trans, Unit, m, k, one, v.Off(n-k, 0), work); err != nil {
					panic(err)
				}

				//              C2 := C2 - W
				for j = 1; j <= k; j++ {
					for i = 1; i <= m; i++ {
						c.Set(i-1, n-k+j-1, c.Get(i-1, n-k+j-1)-work.Get(i-1, j-1))
					}
				}
			}
		}

	} else if storev == 'R' {

		if direct == 'F' {
			//           Let  V =  ( V1  V2 )    (V1: first K columns)
			//           where  V1  is unit upper triangular.
			if side == Left {
				//              Form  H * C  or  H**T * C  where  C = ( C1 )
				//                                                    ( C2 )
				//
				//              W := C**T * V**T  =  (C1**T * V1**T + C2**T * V2**T) (stored in WORK)
				//
				//              W := C1**T
				for j = 1; j <= k; j++ {
					goblas.Dcopy(n, c.Vector(j-1, 0), work.Vector(0, j-1, 1))
				}

				//              W := W * V1**T
				if err = goblas.Dtrmm(Right, Upper, Trans, Unit, n, k, one, v, work); err != nil {
					panic(err)
				}
				if m > k {
					//                 W := W + C2**T * V2**T
					if err = goblas.Dgemm(Trans, Trans, n, k, m-k, one, c.Off(k, 0), v.Off(0, k), one, work); err != nil {
						panic(err)
					}
				}

				//              W := W * T**T  or  W * T
				if err = goblas.Dtrmm(Right, Upper, transt, NonUnit, n, k, one, t, work); err != nil {
					panic(err)
				}

				//              C := C - V**T * W**T
				if m > k {
					//                 C2 := C2 - V2**T * W**T
					if err = goblas.Dgemm(Trans, Trans, m-k, n, k, -one, v.Off(0, k), work, one, c.Off(k, 0)); err != nil {
						panic(err)
					}
				}

				//              W := W * V1
				if err = goblas.Dtrmm(Right, Upper, NoTrans, Unit, n, k, one, v, work); err != nil {
					panic(err)
				}

				//              C1 := C1 - W**T
				for j = 1; j <= k; j++ {
					for i = 1; i <= n; i++ {
						c.Set(j-1, i-1, c.Get(j-1, i-1)-work.Get(i-1, j-1))
					}
				}

			} else if side == Right {
				//              Form  C * H  or  C * H**T  where  C = ( C1  C2 )
				//
				//              W := C * V**T  =  (C1*V1**T + C2*V2**T)  (stored in WORK)
				//
				//              W := C1
				for j = 1; j <= k; j++ {
					goblas.Dcopy(m, c.Vector(0, j-1, 1), work.Vector(0, j-1, 1))
				}

				//              W := W * V1**T
				if err = goblas.Dtrmm(Right, Upper, Trans, Unit, m, k, one, v, work); err != nil {
					panic(err)
				}
				if n > k {
					//                 W := W + C2 * V2**T
					if err = goblas.Dgemm(NoTrans, Trans, m, k, n-k, one, c.Off(0, k), v.Off(0, k), one, work); err != nil {
						panic(err)
					}
				}

				//              W := W * T  or  W * T**T
				if err = goblas.Dtrmm(Right, Upper, trans, NonUnit, m, k, one, t, work); err != nil {
					panic(err)
				}

				//              C := C - W * V
				if n > k {
					//                 C2 := C2 - W * V2
					if err = goblas.Dgemm(NoTrans, NoTrans, m, n-k, k, -one, work, v.Off(0, k), one, c.Off(0, k)); err != nil {
						panic(err)
					}
				}

				//              W := W * V1
				if err = goblas.Dtrmm(Right, Upper, NoTrans, Unit, m, k, one, v, work); err != nil {
					panic(err)
				}

				//              C1 := C1 - W
				for j = 1; j <= k; j++ {
					for i = 1; i <= m; i++ {
						c.Set(i-1, j-1, c.Get(i-1, j-1)-work.Get(i-1, j-1))
					}
				}

			}

		} else {
			//           Let  V =  ( V1  V2 )    (V2: last K columns)
			//           where  V2  is unit lower triangular.
			if side == Left {
				//              Form  H * C  or  H**T * C  where  C = ( C1 )
				//                                                    ( C2 )
				//
				//              W := C**T * V**T  =  (C1**T * V1**T + C2**T * V2**T) (stored in WORK)
				//
				//              W := C2**T
				for j = 1; j <= k; j++ {
					goblas.Dcopy(n, c.Vector(m-k+j-1, 0), work.Vector(0, j-1, 1))
				}

				//              W := W * V2**T
				if err = goblas.Dtrmm(Right, Lower, Trans, Unit, n, k, one, v.Off(0, m-k), work); err != nil {
					panic(err)
				}
				if m > k {
					//                 W := W + C1**T * V1**T
					if err = goblas.Dgemm(Trans, Trans, n, k, m-k, one, c, v, one, work); err != nil {
						panic(err)
					}
				}

				//              W := W * T**T  or  W * T
				if err = goblas.Dtrmm(Right, Lower, transt, NonUnit, n, k, one, t, work); err != nil {
					panic(err)
				}

				//              C := C - V**T * W**T
				if m > k {
					//                 C1 := C1 - V1**T * W**T
					if err = goblas.Dgemm(Trans, Trans, m-k, n, k, -one, v, work, one, c); err != nil {
						panic(err)
					}
				}

				//              W := W * V2
				if err = goblas.Dtrmm(Right, Lower, NoTrans, Unit, n, k, one, v.Off(0, m-k), work); err != nil {
					panic(err)
				}

				//              C2 := C2 - W**T
				for j = 1; j <= k; j++ {
					for i = 1; i <= n; i++ {
						c.Set(m-k+j-1, i-1, c.Get(m-k+j-1, i-1)-work.Get(i-1, j-1))
					}
				}

			} else if side == Right {
				//              Form  C * H  or  C * H'  where  C = ( C1  C2 )
				//
				//              W := C * V**T  =  (C1*V1**T + C2*V2**T)  (stored in WORK)
				//
				//              W := C2
				for j = 1; j <= k; j++ {
					goblas.Dcopy(m, c.Vector(0, n-k+j-1, 1), work.Vector(0, j-1, 1))
				}

				//              W := W * V2**T
				if err = goblas.Dtrmm(Right, Lower, Trans, Unit, m, k, one, v.Off(0, n-k), work); err != nil {
					panic(err)
				}
				if n > k {
					//                 W := W + C1 * V1**T
					if err = goblas.Dgemm(NoTrans, Trans, m, k, n-k, one, c, v, one, work); err != nil {
						panic(err)
					}
				}

				//              W := W * T  or  W * T**T
				if err = goblas.Dtrmm(Right, Lower, trans, NonUnit, m, k, one, t, work); err != nil {
					panic(err)
				}

				//              C := C - W * V
				if n > k {
					//                 C1 := C1 - W * V1
					if err = goblas.Dgemm(NoTrans, NoTrans, m, n-k, k, -one, work, v, one, c); err != nil {
						panic(err)
					}
				}

				//              W := W * V2
				if err = goblas.Dtrmm(Right, Lower, NoTrans, Unit, m, k, one, v.Off(0, n-k), work); err != nil {
					panic(err)
				}

				//              C1 := C1 - W
				for j = 1; j <= k; j++ {
					for i = 1; i <= m; i++ {
						c.Set(i-1, n-k+j-1, c.Get(i-1, n-k+j-1)-work.Get(i-1, j-1))
					}
				}

			}

		}
	}
}
