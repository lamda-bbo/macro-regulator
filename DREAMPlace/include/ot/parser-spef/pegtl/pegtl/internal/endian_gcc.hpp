// Copyright (c) 2017-2018 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#ifndef TAO_PEGTL_INTERNAL_ENDIAN_GCC_HPP
#define TAO_PEGTL_INTERNAL_ENDIAN_GCC_HPP

#include <cstdint>
#include <cstring>

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      namespace internal
      {

#if not defined( __BYTE_ORDER__ )
#error No byte order defined!
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__

         template< unsigned S >
         struct to_and_from_be
         {
            template< typename T >
            static T convert( const T n ) noexcept
            {
               return n;
            }
         };

         template< unsigned S >
         struct to_and_from_le;

         template<>
         struct to_and_from_le< 1 >
         {
            static std::uint8_t convert( const std::uint8_t n ) noexcept
            {
               return n;
            }

            static std::int8_t convert( const std::int8_t n ) noexcept
            {
               return n;
            }
         };

         template<>
         struct to_and_from_le< 2 >
         {
            static std::int16_t convert( const std::int16_t n ) noexcept
            {
               return __builtin_bswap16( n );
            }

            static std::uint16_t convert( const std::uint16_t n ) noexcept
            {
               return __builtin_bswap16( n );
            }
         };

         template<>
         struct to_and_from_le< 4 >
         {
            static float convert( float n ) noexcept
            {
               std::uint32_t u;
               std::memcpy( &u, &n, 4 );
               u = convert( u );
               std::memcpy( &n, &u, 4 );
               return n;
            }

            static std::int32_t convert( const std::int32_t n ) noexcept
            {
               return __builtin_bswap32( n );
            }

            static std::uint32_t convert( const std::uint32_t n ) noexcept
            {
               return __builtin_bswap32( n );
            }
         };

         template<>
         struct to_and_from_le< 8 >
         {
            static double convert( double n ) noexcept
            {
               std::uint64_t u;
               std::memcpy( &u, &n, 8 );
               u = convert( u );
               std::memcpy( &n, &u, 8 );
               return n;
            }

            static std::int64_t convert( const std::int64_t n ) noexcept
            {
               return __builtin_bswap64( n );
            }

            static std::uint64_t convert( const std::uint64_t n ) noexcept
            {
               return __builtin_bswap64( n );
            }
         };

#define TAO_PEGTL_NATIVE_ORDER be
#define TAO_PEGTL_NATIVE_UTF16 utf16_be
#define TAO_PEGTL_NATIVE_UTF32 utf32_be

#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__

         template< unsigned S >
         struct to_and_from_le
         {
            template< typename T >
            static T convert( const T n ) noexcept
            {
               return n;
            }
         };

         template< unsigned S >
         struct to_and_from_be;

         template<>
         struct to_and_from_be< 1 >
         {
            static std::int8_t convert( const std::int8_t n ) noexcept
            {
               return n;
            }

            static std::uint8_t convert( const std::uint8_t n ) noexcept
            {
               return n;
            }
         };

         template<>
         struct to_and_from_be< 2 >
         {
            static std::int16_t convert( const std::int16_t n ) noexcept
            {
               return __builtin_bswap16( n );
            }

            static std::uint16_t convert( const std::uint16_t n ) noexcept
            {
               return __builtin_bswap16( n );
            }
         };

         template<>
         struct to_and_from_be< 4 >
         {
            static float convert( float n ) noexcept
            {
               std::uint32_t u;
               std::memcpy( &u, &n, 4 );
               u = convert( u );
               std::memcpy( &n, &u, 4 );
               return n;
            }

            static std::int32_t convert( const std::int32_t n ) noexcept
            {
               return __builtin_bswap32( n );
            }

            static std::uint32_t convert( const std::uint32_t n ) noexcept
            {
               return __builtin_bswap32( n );
            }
         };

         template<>
         struct to_and_from_be< 8 >
         {
            static double convert( double n ) noexcept
            {
               std::uint64_t u;
               std::memcpy( &u, &n, 8 );
               u = convert( u );
               std::memcpy( &n, &u, 8 );
               return n;
            }

            static std::uint64_t convert( const std::uint64_t n ) noexcept
            {
               return __builtin_bswap64( n );
            }

            static std::int64_t convert( const std::int64_t n ) noexcept
            {
               return __builtin_bswap64( n );
            }
         };

#define TAO_PEGTL_NATIVE_ORDER le
#define TAO_PEGTL_NATIVE_UTF16 utf16_le
#define TAO_PEGTL_NATIVE_UTF32 utf32_le

#else
#error Unknown host byte order!
#endif

      }  // namespace internal

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#endif
